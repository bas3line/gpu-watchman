//! Bounded, metadata-only inspection of local model artifacts.
//!
//! Artifact inspection deliberately proves storage facts only. It never loads
//! tensor payloads, infers runtime residency, or treats checkpoint shards as
//! tensor/pipeline/expert-parallel placement evidence.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ffi::{OsStr, OsString};
use std::fmt::Write as _;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path};

#[cfg(not(unix))]
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use serde::de::{Error as _, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

#[cfg(not(unix))]
use crate::security::open_read_nonblocking;

pub const ARTIFACT_REPORT_VERSION: u32 = 1;

const SAFETENSORS_HEADER_LENGTH_BYTES: u64 = 8;
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100_000_000;
const MAX_TOTAL_HEADER_BYTES: u64 = 256 * 1024 * 1024;
const MAX_INDEX_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DIRECTORY_ENTRIES: usize = 4096;
const MAX_SHARDS: usize = 1024;
const MAX_TENSORS: usize = 500_000;
const MAX_TENSOR_NAME_BYTES: usize = 1024;
const MAX_TOTAL_TENSOR_NAME_BYTES: usize = 64 * 1024 * 1024;
const MAX_SHARD_NAME_BYTES: usize = 255;
const MAX_TENSOR_RANK: usize = 32;
const MAX_DTYPES: usize = 64;
const MAX_DTYPE_NAME_BYTES: usize = 32;
const MAX_METADATA_ENTRIES: usize = 4096;
const MAX_METADATA_BYTES: usize = 8 * 1024 * 1024;
const MAX_INDEX_METADATA_ENTRIES: usize = 64;
const MAX_INDEX_METADATA_KEY_BYTES: usize = 1024;
const INDEX_SUFFIX: &str = ".safetensors.index.json";

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactFormat {
    Safetensors,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactLayout {
    SingleFile,
    ShardedIndex,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ArtifactSummary {
    pub shard_files: u32,
    pub tensor_count: u64,
    pub tensor_elements: u64,
    pub serialized_tensor_bytes: u64,
    pub serialized_shard_file_bytes: u64,
    pub safetensors_header_json_bytes: u64,
    pub safetensors_length_prefix_bytes: u64,
    pub index_file_bytes: Option<u64>,
    pub declared_total_size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ArtifactDtypeSummary {
    pub dtype: String,
    pub tensor_count: u64,
    pub tensor_elements: u64,
    pub serialized_bytes: u64,
    pub shape_payload_bytes_verified: bool,
}

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ArtifactVerification {
    pub regular_shard_files: bool,
    pub final_symlinks_rejected: bool,
    pub directory_descriptors_anchored: bool,
    pub headers_validated: bool,
    pub data_offsets_complete_without_holes: bool,
    pub index_membership_validated: Option<bool>,
    pub declared_total_size_validated: Option<bool>,
    pub shape_payload_bytes_verified_tensors: u64,
    pub shape_payload_bytes_unverified_tensors: u64,
    pub tensor_payload_contents_read: bool,
    pub payload_checksum_validated: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ArtifactReport {
    pub artifact_version: u32,
    pub artifact_format: ArtifactFormat,
    pub layout: ArtifactLayout,
    pub summary: ArtifactSummary,
    pub dtypes: Vec<ArtifactDtypeSummary>,
    pub verification: ArtifactVerification,
    pub caveats: Vec<String>,
}

#[derive(Clone, Copy)]
enum ArtifactSourceKind {
    SingleFile,
    ShardedIndex,
}

struct ArtifactSource {
    kind: ArtifactSourceKind,
    directory: AnchoredDirectory,
    file: File,
}

#[cfg(unix)]
struct AnchoredDirectory {
    file: File,
}

#[cfg(not(unix))]
struct AnchoredDirectory {
    path: PathBuf,
}

#[cfg(unix)]
impl AnchoredDirectory {
    fn open_input(path: &Path) -> Result<Self> {
        Self::open_directory(path, true, "artifact input directory")
    }

    fn open_parent(path: &Path) -> Result<Self> {
        let parent = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        Self::open_directory(parent, false, "artifact parent directory")
    }

    fn open_directory(path: &Path, no_follow: bool, label: &str) -> Result<Self> {
        use rustix::fs::{Mode, OFlags};

        let mut flags = OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC;
        if no_follow {
            flags |= OFlags::NOFOLLOW;
        }
        let descriptor = rustix::fs::open(path, flags, Mode::empty())
            .with_context(|| format!("open {label}"))?;
        Ok(Self {
            file: descriptor.into(),
        })
    }

    fn open_regular(&self, name: &OsStr, label: &str) -> Result<File> {
        use rustix::fs::{Mode, OFlags};

        let descriptor = rustix::fs::openat(
            &self.file,
            name,
            OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC | OFlags::NOFOLLOW,
            Mode::empty(),
        )
        .with_context(|| format!("open {label} through anchored directory"))?;
        let file: File = descriptor.into();
        if !file
            .metadata()
            .with_context(|| format!("inspect opened {label}"))?
            .is_file()
        {
            bail!("{label} is not a regular file");
        }
        Ok(file)
    }

    fn ensure_identity(&self, expected: &std::fs::Metadata, label: &str) -> Result<()> {
        let opened = self
            .file
            .metadata()
            .with_context(|| format!("inspect opened {label}"))?;
        ensure_same_file_identity(expected, &opened, label)
    }

    fn entries(&self) -> Result<Vec<OsString>> {
        use std::os::unix::ffi::OsStringExt;

        let directory = rustix::fs::Dir::read_from(&self.file)
            .context("open anchored artifact directory stream")?;
        let mut names = Vec::new();
        for entry in directory {
            let entry = entry.context("read anchored artifact directory entry")?;
            let bytes = entry.file_name().to_bytes();
            if bytes == b"." || bytes == b".." {
                continue;
            }
            if names.len() >= MAX_DIRECTORY_ENTRIES {
                bail!("artifact directory contains more than {MAX_DIRECTORY_ENTRIES} entries");
            }
            names.push(OsString::from_vec(bytes.to_vec()));
        }
        Ok(names)
    }
}

#[cfg(not(unix))]
impl AnchoredDirectory {
    fn open_input(path: &Path) -> Result<Self> {
        Ok(Self {
            path: path.to_path_buf(),
        })
    }

    fn open_parent(path: &Path) -> Result<Self> {
        Ok(Self {
            path: path
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf(),
        })
    }

    fn open_regular(&self, name: &OsStr, label: &str) -> Result<File> {
        let path = self.path.join(name);
        if std::fs::symlink_metadata(&path)
            .with_context(|| format!("inspect {label}"))?
            .file_type()
            .is_symlink()
        {
            bail!("{label} cannot be a symbolic link");
        }
        let file = open_read_nonblocking(&path, true).with_context(|| format!("open {label}"))?;
        if !file
            .metadata()
            .with_context(|| format!("inspect opened {label}"))?
            .is_file()
        {
            bail!("{label} is not a regular file");
        }
        Ok(file)
    }

    fn entries(&self) -> Result<Vec<OsString>> {
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&self.path).context("read artifact directory")? {
            if names.len() >= MAX_DIRECTORY_ENTRIES {
                bail!("artifact directory contains more than {MAX_DIRECTORY_ENTRIES} entries");
            }
            names.push(entry.context("read artifact directory entry")?.file_name());
        }
        Ok(names)
    }
}

struct HeaderScan {
    tensors: Vec<TensorEvidence>,
    header_json_bytes: u64,
    shard_file_bytes: u64,
}

struct TensorEvidence {
    name: String,
    dtype: String,
    elements: u64,
    serialized_bytes: u64,
    shape_payload_bytes_verified: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TensorDescriptor {
    dtype: DtypeName,
    shape: BoundedShape,
    data_offsets: [u64; 2],
}

struct DtypeName(String);

struct BoundedShape {
    elements: u64,
}

struct HeaderKey(String);

struct TensorName(String);

struct MetadataKey(String);

struct MetadataValueLength(usize);

struct IndexMetadataKey(String);

struct ShardName(String);

#[derive(Clone, Copy)]
struct BoundedOwnedStringVisitor {
    maximum_bytes: usize,
    label: &'static str,
    validator: fn(&str) -> bool,
}

impl BoundedOwnedStringVisitor {
    fn accept<E>(self, value: &str) -> std::result::Result<String, E>
    where
        E: serde::de::Error,
    {
        if value.len() > self.maximum_bytes {
            return Err(E::custom(format!(
                "{} exceeds its {}-byte limit",
                self.label, self.maximum_bytes
            )));
        }
        if !(self.validator)(value) {
            return Err(E::custom(format!(
                "{} violates the safety contract",
                self.label
            )));
        }
        Ok(value.to_owned())
    }
}

impl<'de> Visitor<'de> for BoundedOwnedStringVisitor {
    type Value = String;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} with at most {} UTF-8 bytes",
            self.label, self.maximum_bytes
        )
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.accept(value)
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.accept(value)
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if value.len() > self.maximum_bytes {
            return Err(E::custom(format!(
                "{} exceeds its {}-byte limit",
                self.label, self.maximum_bytes
            )));
        }
        if !(self.validator)(&value) {
            return Err(E::custom(format!(
                "{} violates the safety contract",
                self.label
            )));
        }
        Ok(value)
    }
}

#[derive(Clone, Copy)]
struct BoundedStringLengthVisitor {
    maximum_bytes: usize,
    label: &'static str,
}

impl BoundedStringLengthVisitor {
    fn accept<E>(self, value: &str) -> std::result::Result<usize, E>
    where
        E: serde::de::Error,
    {
        if value.len() > self.maximum_bytes {
            return Err(E::custom(format!(
                "{} exceeds its {}-byte limit",
                self.label, self.maximum_bytes
            )));
        }
        Ok(value.len())
    }
}

impl<'de> Visitor<'de> for BoundedStringLengthVisitor {
    type Value = usize;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} with at most {} UTF-8 bytes",
            self.label, self.maximum_bytes
        )
    }

    fn visit_borrowed_str<E>(self, value: &'de str) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.accept(value)
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.accept(value)
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.accept(&value)
    }
}

fn deserialize_bounded_owned_string<'de, D>(
    deserializer: D,
    maximum_bytes: usize,
    label: &'static str,
    validator: fn(&str) -> bool,
) -> std::result::Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_str(BoundedOwnedStringVisitor {
        maximum_bytes,
        label,
        validator,
    })
}

fn any_string(_: &str) -> bool {
    true
}

fn header_key_is_valid(value: &str) -> bool {
    value == "__metadata__" || tensor_name_is_valid(value)
}

impl<'de> Deserialize<'de> for DtypeName {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_bounded_owned_string(
            deserializer,
            MAX_DTYPE_NAME_BYTES,
            "unsupported safetensors dtype",
            |value| dtype_bits(value).is_some(),
        )
        .map(Self)
    }
}

impl<'de> Deserialize<'de> for HeaderKey {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_bounded_owned_string(
            deserializer,
            MAX_TENSOR_NAME_BYTES,
            "safetensors header key",
            header_key_is_valid,
        )
        .map(Self)
    }
}

impl<'de> Deserialize<'de> for TensorName {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_bounded_owned_string(
            deserializer,
            MAX_TENSOR_NAME_BYTES,
            "safetensors tensor name",
            tensor_name_is_valid,
        )
        .map(Self)
    }
}

impl<'de> Deserialize<'de> for MetadataKey {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_bounded_owned_string(
            deserializer,
            MAX_METADATA_BYTES,
            "safetensors metadata key",
            any_string,
        )
        .map(Self)
    }
}

impl<'de> Deserialize<'de> for MetadataValueLength {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer
            .deserialize_str(BoundedStringLengthVisitor {
                maximum_bytes: MAX_METADATA_BYTES,
                label: "safetensors metadata value",
            })
            .map(Self)
    }
}

impl<'de> Deserialize<'de> for IndexMetadataKey {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_bounded_owned_string(
            deserializer,
            MAX_INDEX_METADATA_KEY_BYTES,
            "safetensors index metadata key",
            any_string,
        )
        .map(Self)
    }
}

impl<'de> Deserialize<'de> for ShardName {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_bounded_owned_string(
            deserializer,
            MAX_SHARD_NAME_BYTES,
            "unsafe shard filename",
            shard_name_is_valid,
        )
        .map(Self)
    }
}

impl<'de> Deserialize<'de> for BoundedShape {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ShapeVisitor;

        impl<'de> Visitor<'de> for ShapeVisitor {
            type Value = BoundedShape;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    formatter,
                    "a tensor shape with at most {MAX_TENSOR_RANK} dimensions"
                )
            }

            fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut dimensions = 0_usize;
                let mut elements = 1_u64;
                while let Some(dimension) = sequence.next_element::<u64>()? {
                    if dimensions >= MAX_TENSOR_RANK {
                        return Err(A::Error::custom("safetensors tensor rank exceeds limit"));
                    }
                    elements = elements.checked_mul(dimension).ok_or_else(|| {
                        A::Error::custom("safetensors tensor element count overflows")
                    })?;
                    dimensions += 1;
                }
                Ok(BoundedShape { elements })
            }
        }

        deserializer.deserialize_seq(ShapeVisitor)
    }
}

struct SafetensorsHeader {
    tensors: Vec<(String, TensorDescriptor)>,
}

impl<'de> Deserialize<'de> for SafetensorsHeader {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct HeaderVisitor;

        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = SafetensorsHeader;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a duplicate-free safetensors header object")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut seen = HashSet::new();
                let mut tensors = Vec::new();
                let mut total_name_bytes = 0_usize;
                while let Some(HeaderKey(name)) = map.next_key()? {
                    if !seen.insert(name.clone()) {
                        return Err(A::Error::custom("duplicate safetensors header key"));
                    }
                    if name == "__metadata__" {
                        let _: UniqueMetadata = map.next_value()?;
                    } else {
                        if tensors.len() >= MAX_TENSORS {
                            return Err(A::Error::custom(
                                "safetensors header tensor count exceeds limit",
                            ));
                        }
                        total_name_bytes = total_name_bytes
                            .checked_add(name.len())
                            .ok_or_else(|| A::Error::custom("tensor name bytes overflow"))?;
                        if total_name_bytes > MAX_TOTAL_TENSOR_NAME_BYTES {
                            return Err(A::Error::custom(
                                "combined safetensors tensor names exceed limit",
                            ));
                        }
                        let descriptor = map.next_value()?;
                        tensors.push((name, descriptor));
                    }
                }
                Ok(SafetensorsHeader { tensors })
            }
        }

        deserializer.deserialize_map(HeaderVisitor)
    }
}

struct UniqueMetadata;

impl<'de> Deserialize<'de> for UniqueMetadata {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MetadataVisitor;

        impl<'de> Visitor<'de> for MetadataVisitor {
            type Value = UniqueMetadata;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a duplicate-free string-to-string metadata object")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut seen = HashSet::new();
                let mut total_bytes = 0_usize;
                while let Some(MetadataKey(key)) = map.next_key()? {
                    if seen.len() >= MAX_METADATA_ENTRIES {
                        return Err(A::Error::custom(
                            "safetensors metadata entry count exceeds limit",
                        ));
                    }
                    let key_len = key.len();
                    if !seen.insert(key) {
                        return Err(A::Error::custom("duplicate safetensors metadata key"));
                    }
                    let MetadataValueLength(value_bytes) = map.next_value()?;
                    total_bytes = total_bytes
                        .checked_add(key_len)
                        .and_then(|total| total.checked_add(value_bytes))
                        .ok_or_else(|| A::Error::custom("safetensors metadata bytes overflow"))?;
                    if total_bytes > MAX_METADATA_BYTES {
                        return Err(A::Error::custom("safetensors metadata exceeds byte limit"));
                    }
                }
                Ok(UniqueMetadata)
            }
        }

        deserializer.deserialize_map(MetadataVisitor)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SafetensorsIndex {
    metadata: IndexMetadata,
    weight_map: UniqueWeightMap,
}

struct IndexMetadata {
    total_size: u64,
}

impl<'de> Deserialize<'de> for IndexMetadata {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MetadataVisitor;

        impl<'de> Visitor<'de> for MetadataVisitor {
            type Value = IndexMetadata;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an index metadata object containing total_size")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut seen = HashSet::new();
                let mut total_size = None;
                while let Some(IndexMetadataKey(key)) = map.next_key()? {
                    if seen.len() >= MAX_INDEX_METADATA_ENTRIES {
                        return Err(A::Error::custom(
                            "safetensors index metadata entry count exceeds limit",
                        ));
                    }
                    if !seen.insert(key.clone()) {
                        return Err(A::Error::custom("duplicate safetensors index metadata key"));
                    }
                    if key == "total_size" {
                        total_size = Some(map.next_value()?);
                    } else {
                        let _: serde::de::IgnoredAny = map.next_value()?;
                    }
                }
                Ok(IndexMetadata {
                    total_size: total_size.ok_or_else(|| A::Error::missing_field("total_size"))?,
                })
            }
        }

        deserializer.deserialize_map(MetadataVisitor)
    }
}

struct UniqueWeightMap(HashMap<String, String>);

impl<'de> Deserialize<'de> for UniqueWeightMap {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct WeightMapVisitor;

        impl<'de> Visitor<'de> for WeightMapVisitor {
            type Value = UniqueWeightMap;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a duplicate-free tensor-to-shard map")
            }

            fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = HashMap::new();
                let mut total_name_bytes = 0_usize;
                while let Some(TensorName(name)) = map.next_key()? {
                    if values.len() >= MAX_TENSORS {
                        return Err(A::Error::custom(
                            "safetensors index tensor count exceeds limit",
                        ));
                    }
                    total_name_bytes = total_name_bytes
                        .checked_add(name.len())
                        .ok_or_else(|| A::Error::custom("index tensor name bytes overflow"))?;
                    if total_name_bytes > MAX_TOTAL_TENSOR_NAME_BYTES {
                        return Err(A::Error::custom("combined index tensor names exceed limit"));
                    }
                    if values.contains_key(&name) {
                        return Err(A::Error::custom("duplicate safetensors index tensor key"));
                    }
                    let ShardName(shard) = map.next_value()?;
                    values.insert(name, shard);
                }
                Ok(UniqueWeightMap(values))
            }
        }

        deserializer.deserialize_map(WeightMapVisitor)
    }
}

#[derive(Default)]
struct DtypeAccumulator {
    tensor_count: u64,
    tensor_elements: u64,
    serialized_bytes: u64,
    verified_tensors: u64,
}

#[derive(Default)]
struct ReportAccumulator {
    shard_files: u64,
    tensor_count: u64,
    tensor_elements: u64,
    serialized_tensor_bytes: u64,
    serialized_shard_file_bytes: u64,
    safetensors_header_json_bytes: u64,
    verified_tensors: u64,
    dtypes: BTreeMap<String, DtypeAccumulator>,
}

impl ReportAccumulator {
    fn add_scan(&mut self, scan: HeaderScan) -> Result<()> {
        self.shard_files = checked_add(self.shard_files, 1, "artifact shard count")?;
        self.serialized_shard_file_bytes = checked_add(
            self.serialized_shard_file_bytes,
            scan.shard_file_bytes,
            "artifact file bytes",
        )?;
        self.safetensors_header_json_bytes = checked_add(
            self.safetensors_header_json_bytes,
            scan.header_json_bytes,
            "artifact header bytes",
        )?;
        if self.safetensors_header_json_bytes > MAX_TOTAL_HEADER_BYTES {
            bail!("combined safetensors headers exceed the safety limit");
        }

        for tensor in scan.tensors {
            self.tensor_count = checked_add(self.tensor_count, 1, "artifact tensor count")?;
            self.tensor_elements = checked_add(
                self.tensor_elements,
                tensor.elements,
                "artifact tensor element count",
            )?;
            self.serialized_tensor_bytes = checked_add(
                self.serialized_tensor_bytes,
                tensor.serialized_bytes,
                "artifact tensor bytes",
            )?;
            if tensor.shape_payload_bytes_verified {
                self.verified_tensors =
                    checked_add(self.verified_tensors, 1, "verified tensor count")?;
            }
            if !self.dtypes.contains_key(&tensor.dtype) && self.dtypes.len() >= MAX_DTYPES {
                bail!("artifact contains more than {MAX_DTYPES} distinct dtypes");
            }
            let dtype = self.dtypes.entry(tensor.dtype).or_default();
            dtype.tensor_count = checked_add(dtype.tensor_count, 1, "dtype tensor count")?;
            dtype.tensor_elements = checked_add(
                dtype.tensor_elements,
                tensor.elements,
                "dtype tensor element count",
            )?;
            dtype.serialized_bytes = checked_add(
                dtype.serialized_bytes,
                tensor.serialized_bytes,
                "dtype tensor bytes",
            )?;
            if tensor.shape_payload_bytes_verified {
                dtype.verified_tensors =
                    checked_add(dtype.verified_tensors, 1, "dtype verified count")?;
            }
        }
        Ok(())
    }

    fn finish(
        self,
        layout: ArtifactLayout,
        index_file_bytes: Option<u64>,
        declared_total_size_bytes: Option<u64>,
    ) -> Result<ArtifactReport> {
        let shard_files = u32::try_from(self.shard_files)
            .map_err(|_| anyhow::anyhow!("artifact shard count is not reportable"))?;
        let length_prefix_bytes = self
            .shard_files
            .checked_mul(SAFETENSORS_HEADER_LENGTH_BYTES)
            .ok_or_else(|| anyhow::anyhow!("artifact length-prefix bytes overflow"))?;
        let unverified_tensors = self
            .tensor_count
            .checked_sub(self.verified_tensors)
            .ok_or_else(|| anyhow::anyhow!("artifact verification counts are inconsistent"))?;
        let dtypes = self
            .dtypes
            .into_iter()
            .map(|(dtype, value)| ArtifactDtypeSummary {
                dtype,
                tensor_count: value.tensor_count,
                tensor_elements: value.tensor_elements,
                serialized_bytes: value.serialized_bytes,
                shape_payload_bytes_verified: value.verified_tensors == value.tensor_count,
            })
            .collect();
        let indexed = matches!(layout, ArtifactLayout::ShardedIndex);
        Ok(ArtifactReport {
            artifact_version: ARTIFACT_REPORT_VERSION,
            artifact_format: ArtifactFormat::Safetensors,
            layout,
            summary: ArtifactSummary {
                shard_files,
                tensor_count: self.tensor_count,
                tensor_elements: self.tensor_elements,
                serialized_tensor_bytes: self.serialized_tensor_bytes,
                serialized_shard_file_bytes: self.serialized_shard_file_bytes,
                safetensors_header_json_bytes: self.safetensors_header_json_bytes,
                safetensors_length_prefix_bytes: length_prefix_bytes,
                index_file_bytes,
                declared_total_size_bytes,
            },
            dtypes,
            verification: ArtifactVerification {
                regular_shard_files: true,
                final_symlinks_rejected: true,
                directory_descriptors_anchored: cfg!(unix),
                headers_validated: true,
                data_offsets_complete_without_holes: true,
                index_membership_validated: indexed.then_some(true),
                declared_total_size_validated: indexed.then_some(true),
                shape_payload_bytes_verified_tensors: self.verified_tensors,
                shape_payload_bytes_unverified_tensors: unverified_tensors,
                tensor_payload_contents_read: false,
                payload_checksum_validated: false,
            },
            caveats: vec![
                "Serialized tensor bytes describe checkpoint storage, not runtime GPU-resident allocation or workspace memory."
                    .to_owned(),
                "Checkpoint shard files are storage containers and do not prove TP, PP, DP, or EP runtime placement."
                    .to_owned(),
                "Tensor payload contents were not read or checksummed; this report validates bounded metadata, offsets, membership, and file lengths."
                    .to_owned(),
            ],
        })
    }
}

/// Inspect a local safetensors file, sharded index, or unambiguous directory.
pub fn inspect(path: &Path) -> Result<ArtifactReport> {
    inspect_source(resolve_source(path)?)
}

fn inspect_source(source: ArtifactSource) -> Result<ArtifactReport> {
    match source.kind {
        ArtifactSourceKind::SingleFile => inspect_single(source.file),
        ArtifactSourceKind::ShardedIndex => inspect_index(&source.directory, source.file),
    }
}

fn inspect_single(file: File) -> Result<ArtifactReport> {
    let scan = scan_safetensors(file)?;
    let mut accumulator = ReportAccumulator::default();
    accumulator.add_scan(scan)?;
    accumulator.finish(ArtifactLayout::SingleFile, None, None)
}

fn inspect_index(directory: &AnchoredDirectory, index_file: File) -> Result<ArtifactReport> {
    let (index_bytes, index_file_bytes) =
        read_bounded_regular(index_file, MAX_INDEX_BYTES, "artifact index")?;
    let index: SafetensorsIndex = serde_json::from_slice(&index_bytes)
        .context("artifact index is not valid bounded safetensors index JSON")?;
    let mut remaining = index.weight_map.0;
    validate_weight_map(&remaining)?;

    let shards: BTreeSet<String> = remaining.values().cloned().collect();
    if shards.is_empty() {
        bail!("artifact index does not reference any shard files");
    }
    if shards.len() > MAX_SHARDS {
        bail!("artifact index references more than {MAX_SHARDS} shard files");
    }

    let mut accumulator = ReportAccumulator::default();
    for shard in shards {
        let shard_file = directory.open_regular(OsStr::new(&shard), "safetensors shard")?;
        let scan = scan_safetensors(shard_file)?;
        for tensor in &scan.tensors {
            match remaining.remove(&tensor.name) {
                Some(expected_shard) if expected_shard == shard => {}
                Some(_) => bail!("artifact index and shard tensor membership disagree"),
                None => bail!("a safetensors shard contains a tensor absent from the index"),
            }
        }
        accumulator.add_scan(scan)?;
    }
    if !remaining.is_empty() {
        bail!("artifact index references tensors absent from its shard headers");
    }
    if accumulator.serialized_tensor_bytes != index.metadata.total_size {
        bail!("artifact index total_size does not match serialized shard tensor bytes");
    }
    accumulator.finish(
        ArtifactLayout::ShardedIndex,
        Some(index_file_bytes),
        Some(index.metadata.total_size),
    )
}

fn resolve_source(path: &Path) -> Result<ArtifactSource> {
    let metadata = std::fs::symlink_metadata(path).context("inspect artifact input")?;
    if metadata.file_type().is_symlink() {
        bail!("artifact input cannot be a symbolic link");
    }
    if metadata.is_file() {
        let name = path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("artifact input has no filename"))?;
        let kind = classify_name(name)?;
        let directory = AnchoredDirectory::open_parent(path)?;
        let file = directory.open_regular(name, "artifact input")?;
        #[cfg(unix)]
        ensure_same_file_identity(
            &metadata,
            &file.metadata().context("inspect opened artifact input")?,
            "artifact input",
        )?;
        return Ok(ArtifactSource {
            kind,
            directory,
            file,
        });
    }
    if !metadata.is_dir() {
        bail!("artifact input must be a regular file or directory");
    }
    let directory = AnchoredDirectory::open_input(path)?;
    #[cfg(unix)]
    directory.ensure_identity(&metadata, "artifact input directory")?;
    resolve_directory(directory)
}

#[cfg(unix)]
fn ensure_same_file_identity(
    expected: &std::fs::Metadata,
    opened: &std::fs::Metadata,
    label: &str,
) -> Result<()> {
    use std::os::unix::fs::MetadataExt;

    if expected.dev() != opened.dev() || expected.ino() != opened.ino() {
        bail!("{label} changed while it was being resolved");
    }
    Ok(())
}

fn resolve_directory(directory: AnchoredDirectory) -> Result<ArtifactSource> {
    let mut indexes = Vec::new();
    let mut singles = Vec::new();
    for name in directory.entries()? {
        let Some(utf8_name) = name.to_str() else {
            continue;
        };
        if utf8_name.ends_with(INDEX_SUFFIX) {
            indexes.push(name);
        } else if Path::new(utf8_name).extension() == Some(OsStr::new("safetensors")) {
            singles.push(name);
        }
    }
    let (kind, name) = match (indexes.len(), singles.len()) {
        (1, _) => (ArtifactSourceKind::ShardedIndex, indexes.remove(0)),
        (0, 1) => (ArtifactSourceKind::SingleFile, singles.remove(0)),
        (0, 0) => bail!("artifact directory contains no safetensors file or index"),
        (0, _) => bail!("artifact directory contains multiple safetensors files without an index"),
        (_, _) => bail!("artifact directory contains multiple safetensors indexes"),
    };
    let file = directory.open_regular(&name, "artifact candidate")?;
    Ok(ArtifactSource {
        kind,
        directory,
        file,
    })
}

fn classify_name(name: &OsStr) -> Result<ArtifactSourceKind> {
    let name = name
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("artifact filename must be valid UTF-8"))?;
    if name.ends_with(INDEX_SUFFIX) {
        Ok(ArtifactSourceKind::ShardedIndex)
    } else if Path::new(name).extension() == Some(OsStr::new("safetensors")) {
        Ok(ArtifactSourceKind::SingleFile)
    } else {
        bail!("artifact file must end in .safetensors or .safetensors.index.json");
    }
}

fn read_bounded_regular(mut file: File, limit: u64, label: &str) -> Result<(Vec<u8>, u64)> {
    let initial_metadata = file.metadata().context("inspect opened artifact file")?;
    let initial_len = initial_metadata.len();
    if initial_len > limit {
        bail!("{label} exceeds the {limit}-byte safety limit");
    }
    let mut bytes = Vec::with_capacity(usize::try_from(initial_len).unwrap_or_default());
    Read::by_ref(&mut file)
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("read {label}"))?;
    if u64::try_from(bytes.len()).unwrap_or(u64::MAX) > limit {
        bail!("{label} exceeds the {limit}-byte safety limit");
    }
    if u64::try_from(bytes.len()).ok() != Some(initial_len) {
        bail!("{label} changed while it was being inspected");
    }
    file.seek(SeekFrom::Start(0))
        .with_context(|| format!("rewind {label} for consistency verification"))?;
    read_exact_match(&mut file, &bytes, label)?;
    let mut trailing = [0_u8; 1];
    if file
        .read(&mut trailing)
        .with_context(|| format!("verify the end of {label}"))?
        != 0
    {
        bail!("{label} changed while it was being inspected");
    }
    let final_metadata = file.metadata().context("reinspect opened artifact file")?;
    ensure_file_snapshot_unchanged(&initial_metadata, &final_metadata, label)?;
    Ok((bytes, initial_len))
}

fn scan_safetensors(mut file: File) -> Result<HeaderScan> {
    let initial_metadata = file
        .metadata()
        .context("inspect opened safetensors shard")?;
    let initial_file_bytes = initial_metadata.len();
    if initial_file_bytes < SAFETENSORS_HEADER_LENGTH_BYTES {
        bail!("safetensors shard is shorter than its header-length prefix");
    }
    let mut length_bytes = [0_u8; 8];
    file.read_exact(&mut length_bytes)
        .context("read safetensors header length")?;
    let header_json_bytes = u64::from_le_bytes(length_bytes);
    if header_json_bytes == 0 || header_json_bytes > MAX_SAFETENSORS_HEADER_BYTES {
        bail!("safetensors header length is outside the bounded safety contract");
    }
    let payload_start = SAFETENSORS_HEADER_LENGTH_BYTES
        .checked_add(header_json_bytes)
        .ok_or_else(|| anyhow::anyhow!("safetensors header offset overflows"))?;
    if payload_start > initial_file_bytes {
        bail!("safetensors header extends beyond the shard file");
    }
    let header_len = usize::try_from(header_json_bytes)
        .map_err(|_| anyhow::anyhow!("safetensors header is not addressable"))?;
    let mut header_bytes = vec![0_u8; header_len];
    file.read_exact(&mut header_bytes)
        .context("read bounded safetensors header")?;
    if header_bytes.first() != Some(&b'{') {
        bail!("safetensors header must begin with a JSON object");
    }
    let header = parse_safetensors_header(&header_bytes)?;
    if header.tensors.len() > MAX_TENSORS {
        bail!("safetensors header contains more than {MAX_TENSORS} tensors");
    }
    let payload_bytes = initial_file_bytes - payload_start;
    let tensors = validate_tensor_descriptors(header.tensors, payload_bytes)?;
    file.seek(SeekFrom::Start(0))
        .context("rewind safetensors header for consistency verification")?;
    read_exact_match(&mut file, &length_bytes, "safetensors header length")?;
    read_exact_match(&mut file, &header_bytes, "safetensors header")?;
    let final_metadata = file
        .metadata()
        .context("reinspect opened safetensors shard")?;
    ensure_file_snapshot_unchanged(&initial_metadata, &final_metadata, "safetensors shard")?;
    Ok(HeaderScan {
        tensors,
        header_json_bytes,
        shard_file_bytes: initial_file_bytes,
    })
}

fn read_exact_match(file: &mut File, expected: &[u8], label: &str) -> Result<()> {
    let mut buffer = [0_u8; 8 * 1024];
    for expected_chunk in expected.chunks(buffer.len()) {
        let actual = &mut buffer[..expected_chunk.len()];
        if file.read_exact(actual).is_err() || actual != expected_chunk {
            bail!("{label} changed while it was being inspected");
        }
    }
    Ok(())
}

fn ensure_file_snapshot_unchanged(
    initial: &std::fs::Metadata,
    final_state: &std::fs::Metadata,
    label: &str,
) -> Result<()> {
    if initial.len() != final_state.len() {
        bail!("{label} changed while it was being inspected");
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        if initial.dev() != final_state.dev()
            || initial.ino() != final_state.ino()
            || initial.mtime() != final_state.mtime()
            || initial.mtime_nsec() != final_state.mtime_nsec()
            || initial.ctime() != final_state.ctime()
            || initial.ctime_nsec() != final_state.ctime_nsec()
        {
            bail!("{label} changed while it was being inspected");
        }
    }

    #[cfg(not(unix))]
    if initial.modified().ok() != final_state.modified().ok() {
        bail!("{label} changed while it was being inspected");
    }

    Ok(())
}

fn parse_safetensors_header(bytes: &[u8]) -> Result<SafetensorsHeader> {
    let mut stream = serde_json::Deserializer::from_slice(bytes).into_iter::<SafetensorsHeader>();
    let header = stream
        .next()
        .ok_or_else(|| anyhow::anyhow!("safetensors header is empty"))?
        .context("safetensors header is not valid bounded JSON")?;
    let consumed = stream.byte_offset();
    if bytes[consumed..].iter().any(|byte| *byte != b' ') {
        bail!("safetensors header may only use ASCII-space trailing padding");
    }
    Ok(header)
}

fn validate_tensor_descriptors(
    descriptors: Vec<(String, TensorDescriptor)>,
    payload_bytes: u64,
) -> Result<Vec<TensorEvidence>> {
    let mut tensors = Vec::with_capacity(descriptors.len());
    for (name, descriptor) in descriptors {
        let elements = descriptor.shape.elements;
        let [start, end] = descriptor.data_offsets;
        if start > end || end > payload_bytes {
            bail!("safetensors tensor data offsets are outside the shard payload");
        }
        let serialized_bytes = end - start;
        let bits = dtype_bits(&descriptor.dtype.0)
            .ok_or_else(|| anyhow::anyhow!("unsupported safetensors dtype"))?;
        let bit_length = elements
            .checked_mul(bits)
            .ok_or_else(|| anyhow::anyhow!("safetensors tensor bit size overflows"))?;
        if !bit_length.is_multiple_of(8) {
            bail!("safetensors sub-byte tensor is not aligned to a complete byte");
        }
        let expected = bit_length / 8;
        if expected != serialized_bytes {
            bail!("safetensors tensor shape, dtype, and data offsets disagree");
        }
        tensors.push((
            start,
            end,
            TensorEvidence {
                name,
                dtype: descriptor.dtype.0,
                elements,
                serialized_bytes,
                shape_payload_bytes_verified: true,
            },
        ));
    }
    tensors.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.name.cmp(&right.2.name))
    });
    let mut cursor = 0_u64;
    for (start, end, _) in &tensors {
        if *start != cursor {
            bail!("safetensors tensor offsets overlap or leave a hole in the payload");
        }
        cursor = *end;
    }
    if cursor != payload_bytes {
        bail!("safetensors tensor offsets do not index the complete payload");
    }
    Ok(tensors.into_iter().map(|(_, _, tensor)| tensor).collect())
}

fn tensor_name_is_valid(name: &str) -> bool {
    !name.is_empty() && name.len() <= MAX_TENSOR_NAME_BYTES && !name.chars().any(char::is_control)
}

fn validate_tensor_name(name: &str) -> Result<()> {
    if !tensor_name_is_valid(name) {
        bail!("safetensors tensor name violates the bounded safety contract");
    }
    Ok(())
}

pub(super) fn dtype_bits(dtype: &str) -> Option<u64> {
    match dtype {
        "F4" => Some(4),
        "F6_E2M3" | "F6_E3M2" => Some(6),
        "BOOL" | "U8" | "I8" | "F8_E5M2" | "F8_E4M3" | "F8_E8M0" | "F8_E4M3FNUZ"
        | "F8_E5M2FNUZ" => Some(8),
        "U16" | "I16" | "F16" | "BF16" => Some(16),
        "U32" | "I32" | "F32" => Some(32),
        "U64" | "I64" | "F64" | "C64" => Some(64),
        _ => None,
    }
}

fn validate_weight_map(weight_map: &HashMap<String, String>) -> Result<()> {
    if weight_map.is_empty() {
        bail!("artifact index weight_map is empty");
    }
    if weight_map.len() > MAX_TENSORS {
        bail!("artifact index contains more than {MAX_TENSORS} tensors");
    }
    let mut total_name_bytes = 0_usize;
    for (tensor, shard) in weight_map {
        validate_tensor_name(tensor)?;
        total_name_bytes = total_name_bytes
            .checked_add(tensor.len())
            .ok_or_else(|| anyhow::anyhow!("artifact tensor names exceed the safety limit"))?;
        if total_name_bytes > MAX_TOTAL_TENSOR_NAME_BYTES {
            bail!("artifact tensor names exceed the combined safety limit");
        }
        validate_shard_name(shard)?;
    }
    Ok(())
}

fn validate_shard_name(name: &str) -> Result<()> {
    if !shard_name_is_valid(name) {
        bail!("artifact index contains an unsafe shard filename");
    }
    Ok(())
}

fn shard_name_is_valid(name: &str) -> bool {
    let path = Path::new(name);
    let mut components = path.components();
    let one_normal_component =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    !name.is_empty()
        && name.len() <= MAX_SHARD_NAME_BYTES
        && name.ends_with(".safetensors")
        && one_normal_component
        && !name
            .bytes()
            .any(|byte| !(byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')))
}

fn checked_add(left: u64, right: u64, label: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| anyhow::anyhow!("{label} overflow"))
}

/// Render the path-free Artifact Report v1 as human-readable text.
pub fn render_text(report: &ArtifactReport) -> String {
    let format = match report.artifact_format {
        ArtifactFormat::Safetensors => "safetensors",
    };
    let layout = match report.layout {
        ArtifactLayout::SingleFile => "single file",
        ArtifactLayout::ShardedIndex => "sharded index",
    };
    let mut output = format!(
        "GPU Watchman artifact report v{}  VERIFIED\n\
         Format      {} | {}\n\
         Shards      {} safetensors file(s)\n\
         Tensors     {} tensor(s) | {} element(s)\n\
         Payload     {}\n\
         Containers  {}\n\
         Headers     {} JSON + {} length prefixes\n",
        report.artifact_version,
        format,
        layout,
        report.summary.shard_files,
        report.summary.tensor_count,
        report.summary.tensor_elements,
        format_bytes(report.summary.serialized_tensor_bytes),
        format_bytes(report.summary.serialized_shard_file_bytes),
        format_bytes(report.summary.safetensors_header_json_bytes),
        format_bytes(report.summary.safetensors_length_prefix_bytes),
    );
    if let Some(index_bytes) = report.summary.index_file_bytes {
        writeln!(
            output,
            "Index       {} | declared payload {}",
            format_bytes(index_bytes),
            format_bytes(report.summary.declared_total_size_bytes.unwrap_or_default())
        )
        .expect("writing to a String cannot fail");
    }
    output.push_str("\nDtypes\n");
    for dtype in &report.dtypes {
        let verification = if dtype.shape_payload_bytes_verified {
            "shape bytes verified"
        } else {
            "offset bytes only"
        };
        writeln!(
            output,
            "  {:<12} {:>8} tensor(s) | {:>14} element(s) | {} | {}",
            dtype.dtype,
            dtype.tensor_count,
            dtype.tensor_elements,
            format_bytes(dtype.serialized_bytes),
            verification,
        )
        .expect("writing to a String cannot fail");
    }
    output.push_str("\nEvidence\n");
    output.push_str("  Headers and complete no-hole data offsets validated.\n");
    if report.verification.index_membership_validated == Some(true) {
        output.push_str(
            "  Index membership and declared total_size validated against every shard.\n",
        );
    }
    output.push_str("  Tensor payload contents were not read or checksummed.\n");
    output.push_str("\nCaveats\n");
    for caveat in &report.caveats {
        output.push_str("  - ");
        output.push_str(caveat);
        output.push('\n');
    }
    output
}

fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;
    if bytes >= GIB {
        format_scaled_bytes(bytes, GIB, "GiB")
    } else if bytes >= MIB {
        format_scaled_bytes(bytes, MIB, "MiB")
    } else if bytes >= KIB {
        format_scaled_bytes(bytes, KIB, "KiB")
    } else {
        format!("{bytes} bytes")
    }
}

fn format_scaled_bytes(bytes: u64, unit: u64, suffix: &str) -> String {
    let hundredths = (u128::from(bytes) * 100 + u128::from(unit / 2)) / u128::from(unit);
    format!(
        "{}.{:02} {suffix} ({bytes} bytes)",
        hundredths / 100,
        hundredths % 100
    )
}

#[cfg(test)]
mod tests {
    use std::fs;

    use serde_json::json;

    use super::*;

    fn write_safetensors(path: &Path, tensors: &[(&str, &str, &[u64], u64)]) {
        let mut offset = 0_u64;
        let mut header = serde_json::Map::new();
        header.insert("__metadata__".to_owned(), json!({"format": "pt"}));
        for (name, dtype, shape, bytes) in tensors {
            header.insert(
                (*name).to_owned(),
                json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + bytes],
                }),
            );
            offset += bytes;
        }
        let mut encoded = serde_json::to_vec(&header).unwrap();
        while !encoded.len().is_multiple_of(8) {
            encoded.push(b' ');
        }
        let mut file = Vec::new();
        file.extend_from_slice(&u64::try_from(encoded.len()).unwrap().to_le_bytes());
        file.extend_from_slice(&encoded);
        file.resize(file.len() + usize::try_from(offset).unwrap(), 0);
        fs::write(path, file).unwrap();
    }

    fn write_raw_safetensors(path: &Path, header: &[u8], payload_bytes: usize) {
        let mut file = Vec::new();
        file.extend_from_slice(&u64::try_from(header.len()).unwrap().to_le_bytes());
        file.extend_from_slice(header);
        file.resize(file.len() + payload_bytes, 0);
        fs::write(path, file).unwrap();
    }

    #[test]
    fn inspects_single_file_without_retaining_names_or_paths() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("private-model.safetensors");
        write_safetensors(
            &path,
            &[
                ("secret.layer.weight", "BF16", &[2, 2], 8),
                ("secret.bias", "F32", &[2], 8),
            ],
        );

        let report = inspect(&path).unwrap();
        assert_eq!(report.artifact_version, ARTIFACT_REPORT_VERSION);
        assert_eq!(report.layout, ArtifactLayout::SingleFile);
        assert_eq!(report.summary.tensor_count, 2);
        assert_eq!(report.summary.tensor_elements, 6);
        assert_eq!(report.summary.serialized_tensor_bytes, 16);
        assert_eq!(report.verification.index_membership_validated, None);
        assert_eq!(
            report.verification.shape_payload_bytes_unverified_tensors,
            0
        );
        let machine = serde_json::to_string(&report).unwrap();
        assert!(!machine.contains("private-model"));
        assert!(!machine.contains("secret.layer"));
        assert!(!render_text(&report).contains("secret"));
    }

    #[test]
    fn verifies_sharded_index_membership_and_total_size() {
        let directory = tempfile::tempdir().unwrap();
        let first = directory.path().join("model-00001-of-00002.safetensors");
        let second = directory.path().join("model-00002-of-00002.safetensors");
        write_safetensors(&first, &[("a.weight", "F16", &[2], 4)]);
        write_safetensors(&second, &[("b.weight", "F32", &[1], 4)]);
        let index = directory.path().join("model.safetensors.index.json");
        fs::write(
            &index,
            serde_json::to_vec(&json!({
                "metadata": {"total_size": 8},
                "weight_map": {
                    "a.weight": "model-00001-of-00002.safetensors",
                    "b.weight": "model-00002-of-00002.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let report = inspect(directory.path()).unwrap();
        assert_eq!(report.layout, ArtifactLayout::ShardedIndex);
        assert_eq!(report.summary.shard_files, 2);
        assert_eq!(report.summary.serialized_tensor_bytes, 8);
        assert_eq!(report.summary.declared_total_size_bytes, Some(8));
        assert_eq!(report.verification.index_membership_validated, Some(true));
        assert_eq!(
            report.verification.declared_total_size_validated,
            Some(true)
        );
    }

    #[test]
    fn rejects_index_membership_and_total_size_mismatches() {
        let directory = tempfile::tempdir().unwrap();
        let shard = directory.path().join("model-00001-of-00001.safetensors");
        write_safetensors(&shard, &[("actual.weight", "F16", &[2], 4)]);
        let index = directory.path().join("model.safetensors.index.json");
        fs::write(
            &index,
            r#"{"metadata":{"total_size":5},"weight_map":{"expected.weight":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();
        let error = inspect(&index).unwrap_err().to_string();
        assert!(error.contains("absent from the index"));

        fs::write(
            &index,
            r#"{"metadata":{"total_size":5},"weight_map":{"actual.weight":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();
        let error = inspect(&index).unwrap_err().to_string();
        assert!(error.contains("total_size"));
    }

    #[test]
    fn rejects_duplicate_header_keys_and_non_string_metadata() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("bad.safetensors");
        let header = br#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut file = Vec::new();
        file.extend_from_slice(&u64::try_from(header.len()).unwrap().to_le_bytes());
        file.extend_from_slice(header);
        file.push(0);
        fs::write(&path, file).unwrap();
        assert!(
            inspect(&path)
                .unwrap_err()
                .to_string()
                .contains("valid bounded JSON")
        );

        let metadata_path = directory.path().join("bad-metadata.safetensors");
        let header = br#"{"__metadata__":{"private":1},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut file = Vec::new();
        file.extend_from_slice(&u64::try_from(header.len()).unwrap().to_le_bytes());
        file.extend_from_slice(header);
        file.push(0);
        fs::write(&metadata_path, file).unwrap();
        assert!(inspect(&metadata_path).is_err());
    }

    #[test]
    fn rejects_holes_overlap_and_known_dtype_shape_mismatch() {
        let directory = tempfile::tempdir().unwrap();
        let gap = directory.path().join("gap.safetensors");
        let header = br#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#;
        let mut file = Vec::new();
        file.extend_from_slice(&u64::try_from(header.len()).unwrap().to_le_bytes());
        file.extend_from_slice(header);
        file.extend_from_slice(&[0, 0]);
        fs::write(&gap, file).unwrap();
        assert!(inspect(&gap).unwrap_err().to_string().contains("hole"));

        let mismatch = directory.path().join("mismatch.safetensors");
        write_safetensors(&mismatch, &[("x", "F32", &[2], 4)]);
        assert!(
            inspect(&mismatch)
                .unwrap_err()
                .to_string()
                .contains("disagree")
        );
    }

    #[test]
    fn accepts_every_supported_dtype_with_exact_bit_accounting() {
        let directory = tempfile::tempdir().unwrap();
        let cases = [
            ("F4", 2, 1),
            ("F6_E2M3", 4, 3),
            ("F6_E3M2", 4, 3),
            ("BOOL", 1, 1),
            ("U8", 1, 1),
            ("I8", 1, 1),
            ("F8_E5M2", 1, 1),
            ("F8_E4M3", 1, 1),
            ("F8_E8M0", 1, 1),
            ("F8_E4M3FNUZ", 1, 1),
            ("F8_E5M2FNUZ", 1, 1),
            ("U16", 1, 2),
            ("I16", 1, 2),
            ("F16", 1, 2),
            ("BF16", 1, 2),
            ("U32", 1, 4),
            ("I32", 1, 4),
            ("F32", 1, 4),
            ("U64", 1, 8),
            ("I64", 1, 8),
            ("F64", 1, 8),
            ("C64", 1, 8),
        ];

        for (index, (dtype, elements, bytes)) in cases.into_iter().enumerate() {
            let path = directory.path().join(format!("dtype-{index}.safetensors"));
            let shape = [elements];
            write_safetensors(&path, &[("x", dtype, &shape, bytes)]);
            let report = inspect(&path).unwrap();
            assert_eq!(report.verification.shape_payload_bytes_verified_tensors, 1);
            assert_eq!(
                report.verification.shape_payload_bytes_unverified_tensors,
                0
            );
            assert_eq!(report.dtypes[0].dtype, dtype);
            assert!(report.dtypes[0].shape_payload_bytes_verified);
        }
    }

    #[test]
    fn rejects_unknown_misaligned_and_incorrectly_sized_dtypes() {
        let directory = tempfile::tempdir().unwrap();

        let unknown = directory.path().join("unknown.safetensors");
        write_safetensors(&unknown, &[("x", "NOT_A_DTYPE", &[1], 1)]);
        assert!(
            format!("{:#}", inspect(&unknown).unwrap_err())
                .contains("unsupported safetensors dtype")
        );

        let misaligned = directory.path().join("misaligned-f4.safetensors");
        write_safetensors(&misaligned, &[("x", "F4", &[3], 1)]);
        assert!(
            inspect(&misaligned)
                .unwrap_err()
                .to_string()
                .contains("complete byte")
        );

        let wrong_c64 = directory.path().join("wrong-c64.safetensors");
        write_safetensors(&wrong_c64, &[("x", "C64", &[1], 1)]);
        assert!(
            inspect(&wrong_c64)
                .unwrap_err()
                .to_string()
                .contains("disagree")
        );
    }

    #[test]
    fn accepts_empty_container_and_rejects_rank_during_decode() {
        let directory = tempfile::tempdir().unwrap();
        let empty = directory.path().join("empty.safetensors");
        write_raw_safetensors(&empty, b"{}", 0);
        let report = inspect(&empty).unwrap();
        assert_eq!(report.summary.tensor_count, 0);
        assert_eq!(report.summary.serialized_tensor_bytes, 0);
        assert!(report.dtypes.is_empty());

        let excessive_rank = directory.path().join("excessive-rank.safetensors");
        let header = serde_json::to_vec(&json!({
            "x": {
                "dtype": "U8",
                "shape": vec![1_u64; MAX_TENSOR_RANK + 1],
                "data_offsets": [0, 1]
            }
        }))
        .unwrap();
        write_raw_safetensors(&excessive_rank, &header, 1);
        assert!(
            format!("{:#}", inspect(&excessive_rank).unwrap_err())
                .contains("tensor rank exceeds limit")
        );
    }

    #[test]
    fn accepts_scalars_zero_sized_tensors_and_maximum_bounded_names() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("shapes.safetensors");
        write_safetensors(
            &path,
            &[
                ("zero.before", "F32", &[0, 7], 0),
                ("scalar", "F16", &[], 2),
                ("zero.after", "U8", &[9, 0], 0),
            ],
        );
        let report = inspect(&path).unwrap();
        assert_eq!(report.summary.tensor_count, 3);
        assert_eq!(report.summary.tensor_elements, 1);
        assert_eq!(report.summary.serialized_tensor_bytes, 2);
        assert_eq!(report.verification.shape_payload_bytes_verified_tensors, 3);

        let maximum_name = "x".repeat(MAX_TENSOR_NAME_BYTES);
        let maximum_path = directory.path().join("maximum-name.safetensors");
        write_safetensors(&maximum_path, &[(&maximum_name, "U8", &[1], 1)]);
        assert_eq!(inspect(&maximum_path).unwrap().summary.tensor_count, 1);

        let excessive_name = "x".repeat(MAX_TENSOR_NAME_BYTES + 1);
        let excessive_path = directory.path().join("excessive-name.safetensors");
        write_safetensors(&excessive_path, &[(&excessive_name, "U8", &[1], 1)]);
        assert!(
            format!("{:#}", inspect(&excessive_path).unwrap_err()).contains("header key exceeds")
        );
    }

    #[test]
    fn enforces_metadata_entry_limit_during_decode() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("metadata-limit.safetensors");
        let mut metadata = serde_json::Map::new();
        for index in 0..MAX_METADATA_ENTRIES {
            metadata.insert(format!("key-{index}"), json!(""));
        }
        let header = serde_json::to_vec(&json!({"__metadata__": metadata.clone()})).unwrap();
        write_raw_safetensors(&path, &header, 0);
        assert_eq!(inspect(&path).unwrap().summary.tensor_count, 0);

        metadata.insert("one-too-many".to_owned(), json!(""));
        let header = serde_json::to_vec(&json!({"__metadata__": metadata})).unwrap();
        write_raw_safetensors(&path, &header, 0);
        assert!(
            format!("{:#}", inspect(&path).unwrap_err())
                .contains("metadata entry count exceeds limit")
        );
    }

    #[test]
    fn rejects_escaped_duplicate_header_keys_and_f6_bit_errors() {
        let directory = tempfile::tempdir().unwrap();
        let duplicate = directory.path().join("escaped-duplicate.safetensors");
        let header = br#"{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"\u0078":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#;
        write_raw_safetensors(&duplicate, header, 1);
        assert!(
            format!("{:#}", inspect(&duplicate).unwrap_err())
                .contains("duplicate safetensors header key")
        );

        let misaligned = directory.path().join("misaligned-f6.safetensors");
        write_safetensors(&misaligned, &[("x", "F6_E2M3", &[1], 1)]);
        assert!(
            inspect(&misaligned)
                .unwrap_err()
                .to_string()
                .contains("complete byte")
        );

        let overflow = directory.path().join("overflow-f4.safetensors");
        let header = serde_json::to_vec(&json!({
            "x": {
                "dtype": "F4",
                "shape": [u64::MAX],
                "data_offsets": [0, 0]
            }
        }))
        .unwrap();
        write_raw_safetensors(&overflow, &header, 0);
        assert!(
            inspect(&overflow)
                .unwrap_err()
                .to_string()
                .contains("bit size overflows")
        );
    }

    #[test]
    fn rejects_unsafe_shard_paths_and_ambiguous_directories() {
        let directory = tempfile::tempdir().unwrap();
        let index = directory.path().join("model.safetensors.index.json");
        fs::write(
            &index,
            r#"{"metadata":{"total_size":1},"weight_map":{"x":"../outside.safetensors"}}"#,
        )
        .unwrap();
        assert!(format!("{:#}", inspect(&index).unwrap_err()).contains("unsafe shard filename"));

        fs::remove_file(&index).unwrap();
        write_safetensors(
            &directory.path().join("one.safetensors"),
            &[("x", "U8", &[1], 1)],
        );
        write_safetensors(
            &directory.path().join("two.safetensors"),
            &[("y", "U8", &[1], 1)],
        );
        assert!(
            inspect(directory.path())
                .unwrap_err()
                .to_string()
                .contains("multiple safetensors files")
        );
    }

    #[test]
    fn rejects_non_space_header_padding() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("newline-padded.safetensors");
        let header = b"{\"x\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}\n";
        let mut file = Vec::new();
        file.extend_from_slice(&u64::try_from(header.len()).unwrap().to_le_bytes());
        file.extend_from_slice(header);
        file.push(0);
        fs::write(&path, file).unwrap();
        assert!(
            inspect(&path)
                .unwrap_err()
                .to_string()
                .contains("ASCII-space trailing padding")
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_final_symlinks_for_inputs_and_shards() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("target.safetensors");
        write_safetensors(&target, &[("x", "U8", &[1], 1)]);
        let link = directory.path().join("link.safetensors");
        symlink(&target, &link).unwrap();
        assert!(
            inspect(&link)
                .unwrap_err()
                .to_string()
                .contains("symbolic link")
        );

        let shard_link = directory.path().join("model-00001-of-00001.safetensors");
        symlink(&target, &shard_link).unwrap();
        let index = directory.path().join("model.safetensors.index.json");
        fs::write(
            &index,
            r#"{"metadata":{"total_size":1},"weight_map":{"x":"model-00001-of-00001.safetensors"}}"#,
        )
        .unwrap();
        assert!(format!("{:#}", inspect(&index).unwrap_err()).contains("symbolic link"));
    }

    #[cfg(unix)]
    #[test]
    fn keeps_shard_resolution_anchored_after_directory_replacement() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("model");
        fs::create_dir(&input).unwrap();
        let shard_name = "model-00001-of-00001.safetensors";
        write_safetensors(&input.join(shard_name), &[("x", "F16", &[2], 4)]);
        fs::write(
            input.join("model.safetensors.index.json"),
            format!(r#"{{"metadata":{{"total_size":4}},"weight_map":{{"x":"{shard_name}"}}}}"#),
        )
        .unwrap();

        let source = resolve_source(&input).unwrap();
        fs::rename(&input, root.path().join("original")).unwrap();
        fs::create_dir(&input).unwrap();
        write_safetensors(&input.join(shard_name), &[("x", "U8", &[1], 1)]);
        fs::write(
            input.join("model.safetensors.index.json"),
            format!(r#"{{"metadata":{{"total_size":1}},"weight_map":{{"x":"{shard_name}"}}}}"#),
        )
        .unwrap();

        let report = inspect_source(source).unwrap();
        assert_eq!(report.summary.tensor_elements, 2);
        assert_eq!(report.summary.serialized_tensor_bytes, 4);
        assert_eq!(report.dtypes[0].dtype, "F16");
        assert!(report.verification.directory_descriptors_anchored);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn ignores_non_utf8_directory_entries() {
        use std::os::unix::ffi::OsStringExt;

        let directory = tempfile::tempdir().unwrap();
        let valid = directory.path().join("valid.safetensors");
        write_safetensors(&valid, &[("x", "U8", &[1], 1)]);
        fs::write(
            directory
                .path()
                .join(OsString::from_vec(b"ignored-\xff.safetensors".to_vec())),
            b"not a checkpoint",
        )
        .unwrap();
        assert_eq!(inspect(directory.path()).unwrap().summary.tensor_count, 1);
    }

    #[cfg(unix)]
    #[test]
    fn detects_same_length_mutation_from_metadata_fingerprint() {
        use std::time::{Duration, SystemTime};

        let directory = tempfile::tempdir().unwrap();
        let mutation = directory.path().join("mutation-check");
        fs::write(&mutation, b"before").unwrap();
        let file = File::open(&mutation).unwrap();
        let initial = file.metadata().unwrap();
        fs::write(&mutation, b"after!").unwrap();
        File::options()
            .write(true)
            .open(&mutation)
            .unwrap()
            .set_times(
                std::fs::FileTimes::new().set_modified(SystemTime::now() + Duration::from_secs(1)),
            )
            .unwrap();
        let final_state = file.metadata().unwrap();
        assert!(ensure_file_snapshot_unchanged(&initial, &final_state, "test").is_err());
    }

    #[test]
    fn inspects_large_sparse_payload_without_reading_tensor_contents() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("sparse.safetensors");
        let payload_bytes = 64_u64 * 1024 * 1024;
        let header = serde_json::to_vec(&json!({
            "x": {
                "dtype": "U8",
                "shape": [payload_bytes],
                "data_offsets": [0, payload_bytes]
            }
        }))
        .unwrap();
        write_raw_safetensors(&path, &header, 0);
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(8 + u64::try_from(header.len()).unwrap() + payload_bytes)
            .unwrap();

        let report = inspect(&path).unwrap();
        assert_eq!(report.summary.serialized_tensor_bytes, payload_bytes);
        assert!(!report.verification.tensor_payload_contents_read);
        assert!(!report.verification.payload_checksum_validated);
    }
}
