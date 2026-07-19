//! Strict TOML operational profiles for monitoring, service, and active canary workflows.
//!
//! The configuration layer performs no hardware or network access. Loading is bounded,
//! profile files may reference secrets only through paths, and callers apply CLI/environment
//! precedence after selecting one validated profile.

mod io;
mod schema;
mod validation;

pub use io::{
    CONFIG_TEMPLATE, LoadedConfig, MAX_CONFIG_BYTES, init_config, load_config, parse_config,
    safe_toml,
};
pub use schema::{
    CONFIG_VERSION, CanaryProfileV1, CanarySloProfileV1, ConfigV1, HumanDuration,
    MonitorCollectionProfileV1, MonitorHealthProfileV1, MonitorInferenceProfileV1,
    MonitorProfileV1, ProfileFailOn, ProfileV1, ServiceProfileV1,
};
pub use validation::{SelectedProfile, select_profile, validate_config, validate_profile_name};

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::time::Duration;

    use super::*;

    fn complete_document() -> &'static str {
        r#"config_version = 1
default_profile = "prod"

[profiles.prod.monitor.collection]
command_timeout = "4s"
gpus = ["0", "GPU-aaaa"]
nvidia_smi = "./bin/nvidia-smi"
collect_xid = false

[profiles.prod.monitor.inference]
urls = ["https://metrics.example/v1?tenant=private"]
token_file = "secrets/probe-token"
timeout = "2s"

[profiles.prod.monitor.health]
fail_on = "critical"
required_sources = ["inventory", "nvidia.topology"]
vram_warning_percent = 88
vram_critical_percent = 97
temperature_warning_c = 80
temperature_critical_c = 89
kv_cache_warning_percent = 80
kv_cache_critical_percent = 94
process_growth_warning_mib = 128

[profiles.prod.service]
listen = "127.0.0.1:9400"
interval = "5s"
history_file = "state/history.ndjson"
freshness = "2m"
api_token_file = "secrets/api-token"
quiet = true

[profiles.prod.canary]
base_url = "https://inference.example/v1?route=private"
model = "served-model"
api_key_file = "secrets/api-key"
prompt_file = "prompts/smoke.txt"
workload_id = "profile-smoke-v1"
max_tokens = 32
count = 10
concurrency = 2
timeout = "20s"
stream = true
allow_insecure_http = false

[profiles.prod.canary.slo]
expect = "gpu-watchman-ok"
max_ttft = "2s"
max_e2e = "10s"
min_output_tokens_per_second = 20
min_success_percent = 100
"#
    }

    #[test]
    fn parses_every_v1_section_and_selects_default_profile() {
        let config = parse_config(complete_document()).unwrap();
        assert_eq!(config.config_version, CONFIG_VERSION);
        let selected = select_profile(&config, None).unwrap().unwrap();
        assert_eq!(selected.name, "prod");
        let canary = selected.profile.canary.as_ref().unwrap();
        assert_eq!(canary.model.as_deref(), Some("served-model"));
        assert_eq!(canary.workload_id.as_deref(), Some("profile-smoke-v1"));
        assert_eq!(canary.timeout.unwrap().0, Duration::from_secs(20));
        assert_eq!(
            canary.slo.as_ref().unwrap().min_success_percent,
            Some(100.0)
        );
    }

    #[test]
    fn rejects_unknown_versions_keys_and_inline_secrets() {
        let future = complete_document().replacen("config_version = 1", "config_version = 2", 1);
        assert!(
            parse_config(&future)
                .unwrap_err()
                .to_string()
                .contains("unsupported config_version 2")
        );

        for invalid in [
            "config_version = 1\nunknown = true\n",
            "config_version = 1\n[profiles.p.monitor.collection]\nunknown = true\n",
            "config_version = 1\n[profiles.p.monitor.inference]\ntoken = \"inline\"\n",
            "config_version = 1\n[profiles.p.service]\napi_token = \"inline\"\n",
            "config_version = 1\n[profiles.p.canary]\napi_key = \"inline\"\n",
            "config_version = 1\n[profiles.p.canary]\nprompt = \"private\"\n",
        ] {
            assert!(
                parse_config(invalid).is_err(),
                "accepted invalid: {invalid}"
            );
        }
    }

    #[test]
    fn parser_and_schema_errors_never_echo_configuration_values() {
        let private = "configuration-value-private";
        for invalid in [
            format!("config_version = 1\nsecret = \"{private}\"\n"),
            format!("config_version = 1\nvalue = \"{private}\n"),
            format!("config_version = 1\n[profiles.p.canary]\ncount = \"{private}\"\n"),
        ] {
            let error = format!("{:#}", parse_config(&invalid).unwrap_err());
            assert!(
                !error.contains(private),
                "configuration value leaked: {error}"
            );
            assert!(error.contains("source excerpt omitted"));
        }
    }

    #[test]
    fn rejects_invalid_names_defaults_thresholds_and_canary_budgets() {
        for invalid in [
            "config_version = 1\ndefault_profile = \"missing\"\n",
            "config_version = 1\n[profiles.\"bad name\"]\n",
            concat!(
                "config_version = 1\n",
                "[profiles.p.monitor.health]\n",
                "vram_warning_percent = 99\n"
            ),
            concat!(
                "config_version = 1\n",
                "[profiles.p.canary]\n",
                "count = 10000\nmax_tokens = 65536\nconcurrency = 64\n"
            ),
            concat!(
                "config_version = 1\n",
                "[profiles.p.canary]\nstream = false\n",
                "[profiles.p.canary.slo]\nmax_ttft = \"1s\"\n"
            ),
            concat!(
                "config_version = 1\n",
                "[profiles.p.canary]\n",
                "base_url = \"http://remote.example/v1\"\n"
            ),
        ] {
            assert!(
                parse_config(invalid).is_err(),
                "accepted invalid: {invalid}"
            );
        }
    }

    #[test]
    fn custom_profile_prompts_require_a_valid_workload_identity() {
        assert!(
            parse_config("config_version = 1\n[profiles.p.canary]\nprompt_file = \"prompt.txt\"\n")
                .is_err()
        );
        assert!(
            parse_config("config_version = 1\n[profiles.p.canary]\nworkload_id = \"orphan-v1\"\n")
                .is_err()
        );
        assert!(
            parse_config(
                "config_version = 1\n[profiles.p.canary]\nprompt_file = \"prompt.txt\"\nworkload_id = \"contains whitespace\"\n"
            )
            .is_err()
        );
        assert!(
            parse_config(
                "config_version = 1\n[profiles.p.canary]\nprompt_file = \"prompt.txt\"\nworkload_id = \"profile-smoke-v1\"\n"
            )
            .is_ok()
        );
    }

    #[test]
    fn remote_transports_and_listeners_require_explicit_profile_opt_in() {
        for invalid in [
            concat!(
                "config_version = 1\n",
                "[profiles.p.monitor.inference]\n",
                "urls = [\"http://metrics.example/metrics\"]\n"
            ),
            concat!(
                "config_version = 1\n",
                "[profiles.p.service]\n",
                "listen = \"0.0.0.0:9400\"\n"
            ),
            concat!(
                "config_version = 1\n",
                "[profiles.p.service]\n",
                "listen = \"not-an-address\"\n"
            ),
        ] {
            assert!(
                parse_config(invalid).is_err(),
                "accepted unsafe profile: {invalid}"
            );
        }

        parse_config(concat!(
            "config_version = 1\n",
            "[profiles.p.monitor.inference]\n",
            "urls = [\"http://metrics.example/metrics\"]\n",
            "allow_insecure_http = true\n",
            "[profiles.p.service]\n",
            "listen = \"0.0.0.0:9400\"\n",
            "allow_remote_listen = true\n"
        ))
        .unwrap();
    }

    #[test]
    fn load_is_bounded_regular_utf8_and_resolves_profile_paths() {
        let directory = tempfile::tempdir().unwrap();
        let config_dir = directory.path().join("config");
        fs::create_dir(&config_dir).unwrap();
        let path = config_dir.join("gpu-watchman.toml");
        fs::write(&path, complete_document()).unwrap();

        let loaded = load_config(&path).unwrap();
        let config_dir = fs::canonicalize(config_dir).unwrap();
        let profile = &loaded.config.profiles["prod"];
        let collection = profile
            .monitor
            .as_ref()
            .unwrap()
            .collection
            .as_ref()
            .unwrap();
        assert_eq!(
            collection.nvidia_smi.as_deref(),
            Some(config_dir.join("bin/nvidia-smi").as_path())
        );
        let monitor = profile.monitor.as_ref().unwrap();
        assert_eq!(
            monitor.inference.as_ref().unwrap().token_file.as_deref(),
            Some(config_dir.join("secrets/probe-token").as_path())
        );
        assert_eq!(
            monitor.health.as_ref().unwrap().required_sources,
            Some(vec![
                "nvidia.inventory".to_owned(),
                "nvidia.topology".to_owned()
            ])
        );
        assert_eq!(
            profile.service.as_ref().unwrap().history_file.as_deref(),
            Some(config_dir.join("state/history.ndjson").as_path())
        );
        assert_eq!(
            profile.canary.as_ref().unwrap().prompt_file.as_deref(),
            Some(config_dir.join("prompts/smoke.txt").as_path())
        );

        let bare_path = config_dir.join("bare.toml");
        fs::write(
            &bare_path,
            "config_version = 1\n[profiles.p.monitor.collection]\nnvidia_smi = \"nvidia-smi\"\n",
        )
        .unwrap();
        let bare = load_config(&bare_path).unwrap();
        assert_eq!(
            bare.config.profiles["p"]
                .monitor
                .as_ref()
                .unwrap()
                .collection
                .as_ref()
                .unwrap()
                .nvidia_smi
                .as_deref(),
            Some(Path::new("nvidia-smi"))
        );

        assert!(load_config(directory.path()).is_err());
        let binary = config_dir.join("binary.toml");
        fs::write(&binary, [0xff, 0xfe]).unwrap();
        assert!(load_config(&binary).is_err());
        let oversized = config_dir.join("oversized.toml");
        let file = fs::File::create(&oversized).unwrap();
        file.set_len(MAX_CONFIG_BYTES + 1).unwrap();
        assert!(load_config(&oversized).is_err());
    }

    #[test]
    fn safe_render_redacts_url_credentials_queries_and_fragments() {
        let mut config = parse_config(complete_document()).unwrap();
        let profile = config.profiles.get_mut("prod").unwrap();
        profile
            .monitor
            .as_mut()
            .unwrap()
            .inference
            .as_mut()
            .unwrap()
            .urls = Some(vec![
            "https://user:password@metrics.example/v1?token=secret#private".to_owned(),
            "not a URL and possibly sensitive".to_owned(),
        ]);
        profile.canary.as_mut().unwrap().base_url =
            Some("https://user:password@inference.example/v1?key=secret".to_owned());

        let rendered = safe_toml(&config).unwrap();
        for private in [
            "user",
            "password",
            "token=",
            "key=",
            "=secret",
            "private",
            "possibly sensitive",
        ] {
            assert!(
                !rendered.contains(private),
                "rendered private value: {private}"
            );
        }
        assert!(rendered.matches("REDACTED").count() >= 2);
        assert!(rendered.contains("<invalid-url>"));
        assert!(rendered.contains("api_key_file"));
    }

    #[test]
    fn init_is_private_valid_and_never_overwrites() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("nested/config.toml");
        init_config(&path).unwrap();
        let original = fs::read_to_string(&path).unwrap();
        assert_eq!(original, CONFIG_TEMPLATE);
        parse_config(&original).unwrap();
        assert!(init_config(&path).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), original);

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn rejects_group_or_world_writable_configuration() {
        use std::os::unix::fs::{PermissionsExt, symlink};

        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("config.toml");
        fs::write(&path, "config_version = 1\n").unwrap();
        let mut permissions = fs::metadata(&path).unwrap().permissions();
        permissions.set_mode(0o666);
        fs::set_permissions(&path, permissions).unwrap();
        let error = load_config(&path).unwrap_err().to_string();
        assert!(error.contains("must not be writable by group or other users"));

        let unsafe_directory = directory.path().join("group-writable");
        fs::create_dir(&unsafe_directory).unwrap();
        let mut permissions = fs::metadata(&unsafe_directory).unwrap().permissions();
        permissions.set_mode(0o770);
        fs::set_permissions(&unsafe_directory, permissions).unwrap();
        let nested = unsafe_directory.join("config.toml");
        fs::write(&nested, "config_version = 1\n").unwrap();
        let error = load_config(&nested).unwrap_err().to_string();
        assert!(error.contains("configuration ancestor"));

        let safe = directory.path().join("safe.toml");
        let link = directory.path().join("linked.toml");
        fs::write(&safe, "config_version = 1\n").unwrap();
        symlink(&safe, &link).unwrap();
        let error = load_config(&link).unwrap_err().to_string();
        assert!(error.contains("must not be a symbolic link"));
    }

    #[cfg(unix)]
    #[test]
    fn relative_paths_use_the_verified_canonical_config_directory() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let real_directory = directory.path().join("real");
        fs::create_dir(&real_directory).unwrap();
        let config_path = real_directory.join("config.toml");
        fs::write(
            &config_path,
            "config_version = 1\n[profiles.local.monitor.collection]\nnvidia_smi = \"bin/nvidia-smi\"\n",
        )
        .unwrap();
        let linked_directory = directory.path().join("linked");
        symlink(&real_directory, &linked_directory).unwrap();

        let loaded = load_config(&linked_directory.join("config.toml")).unwrap();
        let collection = loaded.config.profiles["local"]
            .monitor
            .as_ref()
            .unwrap()
            .collection
            .as_ref()
            .unwrap();
        let real_directory = fs::canonicalize(real_directory).unwrap();
        assert_eq!(loaded.path, real_directory.join("config.toml"));
        assert_eq!(loaded.base_dir, real_directory);
        assert_eq!(
            collection.nvidia_smi.as_deref(),
            Some(real_directory.join("bin/nvidia-smi").as_path())
        );
    }

    #[test]
    fn human_durations_are_string_only_and_round_trip() {
        assert!(parse_config("config_version = 1\n[profiles.p.service]\ninterval = 5\n").is_err());
        let config = parse_config(CONFIG_TEMPLATE).unwrap();
        assert_eq!(
            config.profiles["local"]
                .monitor
                .as_ref()
                .and_then(|monitor| monitor.health.as_ref())
                .and_then(|health| health.fail_on),
            Some(ProfileFailOn::Never)
        );
        let rendered = safe_toml(&config).unwrap();
        let reparsed = parse_config(&rendered).unwrap();
        assert_eq!(config, reparsed);
    }
}
