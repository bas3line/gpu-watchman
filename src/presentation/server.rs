//! Embedded authenticated HTTP API with bounded resource use and distinct
//! liveness and freshness semantics.

use std::collections::{HashMap, VecDeque};
use std::fmt::Write as _;
use std::io::{self, Read, Write};
use std::net::{IpAddr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use serde::Serialize;
use url::{Host, Url};

use super::prometheus;
use crate::domain::Report;
use crate::security::listen_is_loopback;

const WORKER_COUNT: usize = 8;
const CONNECTION_QUEUE_CAPACITY: usize = 8;
const MAX_CONNECTIONS_PER_IP: usize = 2;
const ACCEPT_POLL_INTERVAL: Duration = Duration::from_millis(25);
const READ_TIMEOUT: Duration = Duration::from_millis(750);
const WRITE_TIMEOUT: Duration = Duration::from_secs(2);
const OVERLOAD_WRITE_TIMEOUT: Duration = Duration::from_millis(100);
const MAX_REQUEST_LINE_BYTES: usize = 8 * 1024;
const MAX_HEADER_BYTES: usize = 72 * 1024;
const MAX_HEADER_COUNT: usize = 64;
const MAX_BEARER_TOKEN_BYTES: usize = 64 * 1024;
const MAX_RESPONSE_BODY_BYTES: usize = 2 * 1024 * 1024;
const METRICS_FIXED_UPPER_BOUND: usize = 64 * 1024;
const METRIC_LINE_UPPER_BOUND: usize = 256;
const READ_CHUNK_BYTES: usize = 4 * 1024;

const _: () = assert!(MAX_RESPONSE_BODY_BYTES * WORKER_COUNT <= 16 * 1024 * 1024);

const OVERLOAD_RESPONSE: &[u8] = b"HTTP/1.1 503 Service Unavailable\r\n\
Content-Type: text/plain; charset=utf-8\r\n\
Cache-Control: no-store\r\n\
X-Content-Type-Options: nosniff\r\n\
Connection: close\r\n\
Content-Length: 18\r\n\
\r\n\
server overloaded\n";

#[derive(Debug, Default)]
struct State {
    report: Option<Arc<Report>>,
    last_success: Option<DateTime<Utc>>,
}

#[derive(Debug)]
struct Admission {
    counts: Mutex<HashMap<IpAddr, usize>>,
}

impl Admission {
    fn new() -> Self {
        Self {
            counts: Mutex::new(HashMap::new()),
        }
    }

    fn try_acquire(self: &Arc<Self>, peer: IpAddr) -> Option<AdmissionPermit> {
        let mut counts = self
            .counts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let count = counts.entry(peer).or_default();
        if *count >= MAX_CONNECTIONS_PER_IP {
            return None;
        }
        *count += 1;
        Some(AdmissionPermit {
            admission: Arc::clone(self),
            peer,
        })
    }

    fn release(&self, peer: IpAddr) {
        let mut counts = self
            .counts
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let remove = if let Some(count) = counts.get_mut(&peer) {
            *count = count.saturating_sub(1);
            *count == 0
        } else {
            false
        };
        if remove {
            counts.remove(&peer);
        }
    }
}

#[derive(Debug)]
struct AdmissionPermit {
    admission: Arc<Admission>,
    peer: IpAddr,
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        self.admission.release(self.peer);
    }
}

#[derive(Debug)]
struct Connection {
    stream: TcpStream,
    _permit: AdmissionPermit,
}

#[derive(Debug)]
struct QueueState {
    connections: VecDeque<Connection>,
    closed: bool,
}

#[derive(Debug)]
struct ConnectionQueue {
    state: Mutex<QueueState>,
    available: Condvar,
}

#[derive(Debug)]
struct ActiveState {
    streams: Vec<(u64, TcpStream)>,
    closed: bool,
}

#[derive(Debug)]
struct ActiveConnections {
    next_id: AtomicU64,
    state: Mutex<ActiveState>,
}

impl ActiveConnections {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            state: Mutex::new(ActiveState {
                streams: Vec::with_capacity(WORKER_COUNT),
                closed: false,
            }),
        }
    }

    fn register(&self, stream: &TcpStream) -> io::Result<ActiveConnection<'_>> {
        let stream = stream.try_clone()?;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.closed {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "exporter is shutting down",
            ));
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        state.streams.push((id, stream));
        Ok(ActiveConnection { active: self, id })
    }

    fn unregister(&self, id: u64) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state
            .streams
            .retain(|(connection_id, _)| *connection_id != id);
    }

    fn shutdown_all(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.closed = true;
        for (_, stream) in &state.streams {
            let _ = stream.shutdown(Shutdown::Both);
        }
        state.streams.clear();
    }
}

#[derive(Debug)]
struct ActiveConnection<'a> {
    active: &'a ActiveConnections,
    id: u64,
}

impl Drop for ActiveConnection<'_> {
    fn drop(&mut self) {
        self.active.unregister(self.id);
    }
}

impl ConnectionQueue {
    fn new() -> Self {
        Self {
            state: Mutex::new(QueueState {
                connections: VecDeque::with_capacity(CONNECTION_QUEUE_CAPACITY),
                closed: false,
            }),
            available: Condvar::new(),
        }
    }

    fn try_push(&self, connection: Connection) -> std::result::Result<(), EnqueueError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.closed {
            return Err(EnqueueError::Closed(connection));
        }
        if queue_is_full(state.connections.len()) {
            return Err(EnqueueError::Full(connection));
        }
        state.connections.push_back(connection);
        self.available.notify_one();
        Ok(())
    }

    fn pop(&self) -> Option<Connection> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        loop {
            if let Some(stream) = state.connections.pop_front() {
                return Some(stream);
            }
            if state.closed {
                return None;
            }
            state = self
                .available
                .wait(state)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }

    fn close(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.closed = true;
        state.connections.clear();
        self.available.notify_all();
    }
}

fn queue_is_full(connection_count: usize) -> bool {
    connection_count >= CONNECTION_QUEUE_CAPACITY
}

#[derive(Debug)]
enum EnqueueError {
    Full(Connection),
    Closed(Connection),
}

#[derive(Debug)]
pub struct Exporter {
    state: Arc<RwLock<State>>,
    stop: Arc<AtomicBool>,
    queue: Arc<ConnectionQueue>,
    active: Arc<ActiveConnections>,
    acceptor: Option<JoinHandle<()>>,
    workers: Vec<JoinHandle<()>>,
    address: String,
}

impl Exporter {
    pub fn start(address: &str, freshness: Duration, bearer_token: Option<String>) -> Result<Self> {
        let bearer_token = bearer_token.context(
            "exporter bearer authentication is required; unauthenticated loopback must be explicitly enabled",
        )?;
        Self::start_server(address, freshness, Some(bearer_token))
    }

    pub fn start_unauthenticated_loopback(address: &str, freshness: Duration) -> Result<Self> {
        Self::start_server(address, freshness, None)
    }

    fn start_server(
        address: &str,
        freshness: Duration,
        bearer_token: Option<String>,
    ) -> Result<Self> {
        let declared_loopback = listen_is_loopback(address).map_err(anyhow::Error::msg)?;
        if !declared_loopback && bearer_token.is_none() {
            bail!("unauthenticated exporter listeners must use a loopback address");
        }
        if let Some(token) = bearer_token.as_deref() {
            validate_bearer_token(token)?;
        }

        let listener =
            TcpListener::bind(address).with_context(|| format!("listen on {address}"))?;
        let bound_address = listener
            .local_addr()
            .context("read exporter listen address")?;
        if !bound_address.ip().is_loopback() && bearer_token.is_none() {
            bail!("unauthenticated exporter listeners must resolve to a loopback address");
        }
        listener
            .set_nonblocking(true)
            .context("configure exporter listener")?;

        let state = Arc::new(RwLock::new(State::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let queue = Arc::new(ConnectionQueue::new());
        let active = Arc::new(ActiveConnections::new());
        let admission = Arc::new(Admission::new());
        let token = bearer_token.map(Arc::<str>::from);
        let workers = spawn_workers(
            &queue,
            &active,
            &state,
            freshness,
            token.as_deref(),
            bound_address,
        )?;

        let acceptor_stop = Arc::clone(&stop);
        let acceptor_queue = Arc::clone(&queue);
        let acceptor_admission = Arc::clone(&admission);
        let acceptor = match std::thread::Builder::new()
            .name("gpu-watchman-http-acceptor".to_owned())
            .spawn(move || {
                accept_loop(
                    &listener,
                    &acceptor_queue,
                    &acceptor_admission,
                    &acceptor_stop,
                );
            }) {
            Ok(acceptor) => acceptor,
            Err(error) => {
                queue.close();
                join_all(workers);
                return Err(error).context("start exporter acceptor thread");
            }
        };

        Ok(Self {
            state,
            stop,
            queue,
            active,
            acceptor: Some(acceptor),
            workers,
            address: bound_address.to_string(),
        })
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    pub fn set_report(&self, report: &Report) {
        if let Ok(mut state) = self.state.write() {
            state.last_success = Some(report.collected_at);
            state.report = Some(Arc::new(report.clone()));
        }
    }
}

impl Drop for Exporter {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.queue.close();
        self.active.shutdown_all();
        if let Some(acceptor) = self.acceptor.take() {
            let _ = acceptor.join();
        }
        join_all(std::mem::take(&mut self.workers));
    }
}

fn spawn_workers(
    queue: &Arc<ConnectionQueue>,
    active: &Arc<ActiveConnections>,
    state: &Arc<RwLock<State>>,
    freshness: Duration,
    bearer_token: Option<&str>,
    bound_address: SocketAddr,
) -> Result<Vec<JoinHandle<()>>> {
    let token = bearer_token.map(Arc::<str>::from);
    let mut workers = Vec::with_capacity(WORKER_COUNT);
    for index in 0..WORKER_COUNT {
        let worker_queue = Arc::clone(queue);
        let worker_active = Arc::clone(active);
        let worker_state = Arc::clone(state);
        let worker_token = token.clone();
        match std::thread::Builder::new()
            .name(format!("gpu-watchman-http-{index}"))
            .spawn(move || {
                worker_loop(
                    &worker_queue,
                    &worker_active,
                    &worker_state,
                    freshness,
                    worker_token.as_deref(),
                    bound_address,
                );
            }) {
            Ok(worker) => workers.push(worker),
            Err(error) => {
                queue.close();
                join_all(workers);
                return Err(error).context("start exporter worker thread");
            }
        }
    }
    Ok(workers)
}

fn join_all(workers: Vec<JoinHandle<()>>) {
    for worker in workers {
        let _ = worker.join();
    }
}

fn accept_loop(
    listener: &TcpListener,
    queue: &ConnectionQueue,
    admission: &Arc<Admission>,
    stop: &AtomicBool,
) {
    while !stop.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((mut stream, peer)) => {
                if configure_stream(&stream).is_err() {
                    continue;
                }
                let Some(permit) = admission.try_acquire(peer.ip()) else {
                    reject_overload(&mut stream);
                    continue;
                };
                let connection = Connection {
                    stream,
                    _permit: permit,
                };
                match queue.try_push(connection) {
                    Ok(()) => {}
                    Err(EnqueueError::Full(mut connection)) => {
                        reject_overload(&mut connection.stream);
                    }
                    Err(EnqueueError::Closed(connection)) => {
                        drop(connection);
                        break;
                    }
                }
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                std::thread::park_timeout(ACCEPT_POLL_INTERVAL);
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    queue.close();
}

fn configure_stream(stream: &TcpStream) -> io::Result<()> {
    stream.set_read_timeout(Some(READ_TIMEOUT))?;
    stream.set_write_timeout(Some(WRITE_TIMEOUT))?;
    stream.set_nodelay(true)
}

fn reject_overload(stream: &mut TcpStream) {
    let _ = write_parts_with_timeout(stream, &[OVERLOAD_RESPONSE], OVERLOAD_WRITE_TIMEOUT);
    let _ = stream.shutdown(Shutdown::Both);
}

fn worker_loop(
    queue: &ConnectionQueue,
    active: &ActiveConnections,
    state: &RwLock<State>,
    freshness: Duration,
    bearer_token: Option<&str>,
    bound_address: SocketAddr,
) {
    while let Some(mut connection) = queue.pop() {
        let Ok(_active_connection) = active.register(&connection.stream) else {
            let _ = connection.stream.shutdown(Shutdown::Both);
            continue;
        };
        handle_connection(
            &mut connection.stream,
            state,
            freshness,
            bearer_token,
            bound_address,
        );
    }
}

fn handle_connection(
    stream: &mut TcpStream,
    state: &RwLock<State>,
    freshness: Duration,
    bearer_token: Option<&str>,
    bound_address: SocketAddr,
) {
    let response = match read_request(stream) {
        Ok(request) => Some(route_request(
            &request,
            state,
            freshness,
            bearer_token,
            bound_address,
        )),
        Err(error) => error.response(),
    };
    if let Some(response) = response {
        let _ = write_response(stream, &response);
    }
    let _ = stream.shutdown(Shutdown::Both);
}

trait RequestRead: Read {
    fn set_request_timeout(&self, timeout: Duration) -> io::Result<()>;
}

impl RequestRead for TcpStream {
    fn set_request_timeout(&self, timeout: Duration) -> io::Result<()> {
        self.set_read_timeout(Some(timeout))
    }
}

fn read_request(reader: &mut impl RequestRead) -> std::result::Result<ParsedRequest, RequestError> {
    let deadline = Instant::now() + READ_TIMEOUT;
    let mut bytes = Vec::with_capacity(READ_CHUNK_BYTES);
    let mut chunk = [0_u8; READ_CHUNK_BYTES];
    loop {
        if let Some(end) = find_header_end(&bytes) {
            return parse_request(&bytes[..end]);
        }
        validate_partial_size(&bytes)?;
        let read_capacity = remaining_request_capacity(&bytes)?;
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or(RequestError::Timeout)?;
        reader
            .set_request_timeout(remaining)
            .map_err(|_| RequestError::Connection)?;
        match reader.read(&mut chunk[..read_capacity.min(READ_CHUNK_BYTES)]) {
            Ok(0) => return Err(RequestError::BadRequest),
            Ok(read) => bytes.extend_from_slice(&chunk[..read]),
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                ) =>
            {
                return Err(RequestError::Timeout);
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => return Err(RequestError::Connection),
        }
    }
}

fn remaining_request_capacity(bytes: &[u8]) -> std::result::Result<usize, RequestError> {
    let (limit, error) = if let Some(request_line_end) = find_crlf(bytes) {
        (
            request_line_end + 2 + MAX_HEADER_BYTES,
            RequestError::HeadersTooLarge,
        )
    } else {
        (MAX_REQUEST_LINE_BYTES + 2, RequestError::RequestLineTooLong)
    };
    limit
        .checked_sub(bytes.len())
        .filter(|remaining| *remaining > 0)
        .ok_or(error)
}

fn validate_partial_size(bytes: &[u8]) -> std::result::Result<(), RequestError> {
    if let Some(request_line_end) = find_crlf(bytes) {
        if request_line_end > MAX_REQUEST_LINE_BYTES {
            return Err(RequestError::RequestLineTooLong);
        }
        if bytes.len().saturating_sub(request_line_end + 2) > MAX_HEADER_BYTES {
            return Err(RequestError::HeadersTooLarge);
        }
    } else if bytes.len() > MAX_REQUEST_LINE_BYTES {
        let waiting_for_line_feed = bytes.len() == MAX_REQUEST_LINE_BYTES + 1
            && bytes.last().is_some_and(|byte| *byte == b'\r');
        if !waiting_for_line_feed {
            return Err(RequestError::RequestLineTooLong);
        }
    }
    Ok(())
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
}

fn find_crlf(bytes: &[u8]) -> Option<usize> {
    bytes.windows(2).position(|window| window == b"\r\n")
}

#[derive(Debug, PartialEq, Eq)]
enum RequestError {
    BadRequest,
    RequestLineTooLong,
    HeadersTooLarge,
    TooManyHeaders,
    VersionNotSupported,
    Timeout,
    Connection,
}

impl RequestError {
    fn response(&self) -> Option<HttpResponse> {
        match self {
            Self::BadRequest => Some(text_response(
                400,
                "bad request\n",
                "text/plain; charset=utf-8",
            )),
            Self::RequestLineTooLong => Some(text_response(
                414,
                "request line too long\n",
                "text/plain; charset=utf-8",
            )),
            Self::HeadersTooLarge | Self::TooManyHeaders => Some(text_response(
                431,
                "request headers too large\n",
                "text/plain; charset=utf-8",
            )),
            Self::VersionNotSupported => Some(text_response(
                505,
                "http version not supported\n",
                "text/plain; charset=utf-8",
            )),
            Self::Timeout => Some(text_response(
                408,
                "request timeout\n",
                "text/plain; charset=utf-8",
            )),
            Self::Connection => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum HeaderValue {
    Missing,
    Single(String),
    Duplicate,
}

#[derive(Debug, PartialEq, Eq)]
struct ParsedRequest {
    method: String,
    route: String,
    host: HeaderValue,
    authorization: HeaderValue,
}

fn parse_request(bytes: &[u8]) -> std::result::Result<ParsedRequest, RequestError> {
    let header_end = find_header_end(bytes).ok_or(RequestError::BadRequest)?;
    let request_line_end = find_crlf(&bytes[..header_end]).ok_or(RequestError::BadRequest)?;
    if request_line_end > MAX_REQUEST_LINE_BYTES {
        return Err(RequestError::RequestLineTooLong);
    }
    if header_end.saturating_sub(request_line_end + 2) > MAX_HEADER_BYTES {
        return Err(RequestError::HeadersTooLarge);
    }

    let request_line =
        std::str::from_utf8(&bytes[..request_line_end]).map_err(|_| RequestError::BadRequest)?;
    let mut parts = request_line.split(' ');
    let method = parts.next().ok_or(RequestError::BadRequest)?;
    let target = parts.next().ok_or(RequestError::BadRequest)?;
    let version = parts.next().ok_or(RequestError::BadRequest)?;
    if parts.next().is_some() || method.is_empty() || !valid_token(method.as_bytes()) {
        return Err(RequestError::BadRequest);
    }
    if !matches!(version, "HTTP/1.0" | "HTTP/1.1") {
        return Err(RequestError::VersionNotSupported);
    }
    if !target.starts_with('/')
        || target.contains('#')
        || !target.bytes().all(|byte| (0x21..=0x7e).contains(&byte))
    {
        return Err(RequestError::BadRequest);
    }

    let mut cursor = request_line_end + 2;
    let mut header_count = 0_usize;
    let mut host = HeaderValue::Missing;
    let mut authorization = HeaderValue::Missing;
    loop {
        let line_end = find_crlf(&bytes[cursor..header_end])
            .map(|relative| cursor + relative)
            .ok_or(RequestError::BadRequest)?;
        if line_end == cursor {
            break;
        }
        header_count += 1;
        if header_count > MAX_HEADER_COUNT {
            return Err(RequestError::TooManyHeaders);
        }
        let line = &bytes[cursor..line_end];
        let colon = line
            .iter()
            .position(|byte| *byte == b':')
            .ok_or(RequestError::BadRequest)?;
        let name = &line[..colon];
        if name.is_empty() || !valid_token(name) {
            return Err(RequestError::BadRequest);
        }
        let value = trim_optional_whitespace(&line[colon + 1..]);
        if value
            .iter()
            .any(|byte| (*byte < 0x20 && *byte != b'\t') || *byte == 0x7f)
        {
            return Err(RequestError::BadRequest);
        }
        if name.eq_ignore_ascii_case(b"host") {
            observe_header(&mut host, value)?;
        } else if name.eq_ignore_ascii_case(b"authorization") {
            observe_header(&mut authorization, value)?;
        }
        cursor = line_end + 2;
    }
    if matches!(host, HeaderValue::Single(ref value) if parse_host_authority(value).is_none())
        || matches!(host, HeaderValue::Duplicate)
        || (version == "HTTP/1.1" && !matches!(host, HeaderValue::Single(_)))
    {
        return Err(RequestError::BadRequest);
    }

    Ok(ParsedRequest {
        method: method.to_owned(),
        route: target.split('?').next().unwrap_or_default().to_owned(),
        host,
        authorization,
    })
}

fn observe_header(header: &mut HeaderValue, value: &[u8]) -> std::result::Result<(), RequestError> {
    *header = match header {
        HeaderValue::Missing => HeaderValue::Single(
            std::str::from_utf8(value)
                .map_err(|_| RequestError::BadRequest)?
                .to_owned(),
        ),
        HeaderValue::Single(_) | HeaderValue::Duplicate => HeaderValue::Duplicate,
    };
    Ok(())
}

fn valid_token(value: &[u8]) -> bool {
    value.iter().all(|byte| {
        byte.is_ascii_alphanumeric()
            || matches!(
                byte,
                b'!' | b'#'
                    | b'$'
                    | b'%'
                    | b'&'
                    | b'\''
                    | b'*'
                    | b'+'
                    | b'-'
                    | b'.'
                    | b'^'
                    | b'_'
                    | b'`'
                    | b'|'
                    | b'~'
            )
    })
}

fn trim_optional_whitespace(mut value: &[u8]) -> &[u8] {
    while matches!(value.first(), Some(b' ' | b'\t')) {
        value = &value[1..];
    }
    while matches!(value.last(), Some(b' ' | b'\t')) {
        value = &value[..value.len() - 1];
    }
    value
}

fn route_request(
    request: &ParsedRequest,
    state: &RwLock<State>,
    freshness: Duration,
    bearer_token: Option<&str>,
    bound_address: SocketAddr,
) -> HttpResponse {
    if bearer_token.is_none() && !loopback_host_authorized(&request.host, bound_address) {
        return text_response(400, "invalid host\n", "text/plain; charset=utf-8");
    }
    if !public_probe_route(&request.route)
        && let Some(token) = bearer_token
        && !authorized(request, token)
    {
        return text_response(401, "unauthorized\n", "text/plain; charset=utf-8")
            .with_header("WWW-Authenticate", "Bearer");
    }
    if request.method != "GET" {
        return text_response(405, "method not allowed\n", "text/plain; charset=utf-8")
            .with_header("Allow", "GET");
    }
    if request.route == "/livez" {
        return json_response(
            200,
            &serde_json::json!({"status": "ok", "service": "gpu-watchman"}),
        );
    }

    let snapshot = state.read().ok();
    let report = snapshot.as_ref().and_then(|state| state.report.clone());
    let last_success = snapshot.as_ref().and_then(|state| state.last_success);
    drop(snapshot);

    match request.route.as_str() {
        "/metrics" => metrics_response(report.as_deref(), last_success),
        "/healthz" => health_response(report.as_deref(), last_success, freshness),
        "/api/v1/report" => match report {
            Some(report) => json_response(200, report.as_ref()),
            None => json_response(
                503,
                &serde_json::json!({"error": "no report has been collected"}),
            ),
        },
        "/" => json_response(
            200,
            &serde_json::json!({
                "name": "gpu-watchman",
                "version": env!("CARGO_PKG_VERSION"),
                "endpoints": ["/livez", "/metrics", "/healthz", "/api/v1/report"]
            }),
        ),
        _ => text_response(404, "not found\n", "text/plain; charset=utf-8"),
    }
}

fn public_probe_route(route: &str) -> bool {
    matches!(route, "/livez" | "/healthz")
}

fn validate_bearer_token(token: &str) -> Result<()> {
    if token.is_empty() || token.len() > MAX_BEARER_TOKEN_BYTES {
        bail!("exporter bearer token must contain 1 to {MAX_BEARER_TOKEN_BYTES} bytes");
    }
    if !token.bytes().all(|byte| (0x21..=0x7e).contains(&byte)) {
        bail!("exporter bearer token must contain visible ASCII without whitespace");
    }
    Ok(())
}

fn authorized(request: &ParsedRequest, token: &str) -> bool {
    match &request.authorization {
        HeaderValue::Single(value) => bearer_authorized(Some(value), token),
        HeaderValue::Missing | HeaderValue::Duplicate => false,
    }
}

fn loopback_host_authorized(host: &HeaderValue, bound_address: SocketAddr) -> bool {
    let HeaderValue::Single(authority) = host else {
        return false;
    };
    let Some(url) = parse_host_authority(authority) else {
        return false;
    };
    if url.port_or_known_default() != Some(bound_address.port()) {
        return false;
    }
    match url.host() {
        Some(Host::Domain(host)) => host.eq_ignore_ascii_case("localhost"),
        Some(Host::Ipv4(address)) => bound_address.ip() == IpAddr::V4(address),
        Some(Host::Ipv6(address)) => bound_address.ip() == IpAddr::V6(address),
        None => false,
    }
}

fn parse_host_authority(authority: &str) -> Option<Url> {
    let url = Url::parse(&format!("http://{authority}")).ok()?;
    if url.host().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return None;
    }
    Some(url)
}

fn bearer_authorized(value: Option<&str>, token: &str) -> bool {
    let Some((scheme, candidate)) = value.and_then(|value| value.split_once(' ')) else {
        return false;
    };
    scheme.eq_ignore_ascii_case("Bearer")
        && constant_time_eq(candidate.as_bytes(), token.as_bytes())
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let maximum = left.len().max(right.len());
    let mut difference = left.len() ^ right.len();
    for index in 0..maximum {
        difference |= usize::from(left.get(index).copied().unwrap_or_default())
            ^ usize::from(right.get(index).copied().unwrap_or_default());
    }
    difference == 0
}

#[allow(clippy::cast_precision_loss)]
fn health_response(
    report: Option<&Report>,
    last_success: Option<DateTime<Utc>>,
    freshness: Duration,
) -> HttpResponse {
    let (Some(report), Some(last_success)) = (report, last_success) else {
        return json_response(
            503,
            &HealthResponse {
                status: "unavailable",
                report_status: None,
                last_success: None,
                report_age_seconds: None,
                freshness_threshold_seconds: freshness.as_secs_f64(),
                reason: Some("no report has been collected"),
            },
        );
    };
    let age = Utc::now()
        .signed_duration_since(last_success)
        .num_milliseconds()
        .max(0) as f64
        / 1_000.0;
    let stale = age > freshness.as_secs_f64();
    json_response(
        if stale { 503 } else { 200 },
        &HealthResponse {
            status: if stale { "stale" } else { "ok" },
            report_status: Some(&report.status),
            last_success: Some(last_success),
            report_age_seconds: Some(age),
            freshness_threshold_seconds: freshness.as_secs_f64(),
            reason: stale.then_some("last successful report is too old"),
        },
    )
}

#[derive(Serialize)]
struct HealthResponse<'a> {
    status: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    report_status: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_success: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    report_age_seconds: Option<f64>,
    freshness_threshold_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<&'a str>,
}

#[derive(Debug)]
struct HttpResponse {
    status: u16,
    content_type: &'static str,
    body: Vec<u8>,
    cache_control: &'static str,
    headers: Vec<(&'static str, &'static str)>,
}

impl HttpResponse {
    fn with_header(mut self, name: &'static str, value: &'static str) -> Self {
        self.headers.push((name, value));
        self
    }
}

fn json_response(status: u16, value: &impl Serialize) -> HttpResponse {
    let mut writer = LimitedWriter::new(MAX_RESPONSE_BODY_BYTES);
    match serde_json::to_writer(&mut writer, value) {
        Ok(()) if writer.write_all(b"\n").is_ok() => HttpResponse {
            status,
            content_type: "application/json; charset=utf-8",
            body: writer.into_inner(),
            cache_control: "no-store",
            headers: Vec::new(),
        },
        _ if writer.exceeded() => response_too_large(),
        _ => fixed_json_error(500, "could not encode response"),
    }
}

fn metrics_response(report: Option<&Report>, last_success: Option<DateTime<Utc>>) -> HttpResponse {
    if report.is_some_and(|report| {
        prometheus_size_upper_bound(report).is_none_or(|size| size > MAX_RESPONSE_BODY_BYTES)
    }) {
        return response_too_large();
    }
    text_response_owned(
        200,
        prometheus::encode(report, last_success).into_bytes(),
        "text/plain; version=0.0.4; charset=utf-8",
    )
}

fn prometheus_size_upper_bound(report: &Report) -> Option<usize> {
    let mut total = METRICS_FIXED_UPPER_BOUND;
    add_metric_estimate(&mut total, 1, escaped_upper_bound(report.status.len())?)?;
    if total > MAX_RESPONSE_BODY_BYTES {
        return Some(total);
    }
    for source in &report.sources {
        add_metric_estimate(
            &mut total,
            4,
            escaped_upper_bound(checked_sum(&[
                source.name.len(),
                source.state.as_str().len(),
            ])?)?,
        )?;
        if total > MAX_RESPONSE_BODY_BYTES {
            return Some(total);
        }
    }
    for gpu in &report.gpus {
        let gpu_labels = escaped_upper_bound(checked_sum(&[
            gpu.index.to_string().len(),
            gpu.uuid.len(),
            gpu.name.len(),
        ])?)?;
        add_metric_estimate(&mut total, 16, gpu_labels)?;
        if total > MAX_RESPONSE_BODY_BYTES {
            return Some(total);
        }
        for process in &gpu.processes {
            let process_labels = gpu_labels.checked_add(escaped_upper_bound(checked_sum(&[
                process.pid.to_string().len(),
                process.name.len(),
                process.owner.len(),
            ])?)?)?;
            add_metric_estimate(&mut total, 1, process_labels)?;
            if total > MAX_RESPONSE_BODY_BYTES {
                return Some(total);
            }
        }
    }
    for endpoint in &report.endpoints {
        let runtime = if endpoint.runtime.is_empty() {
            "unknown"
        } else {
            &endpoint.runtime
        };
        let endpoint_labels =
            escaped_upper_bound(checked_sum(&[endpoint.url.len(), runtime.len()])?)?;
        // Base endpoint gauges plus four interval histograms, each with one
        // sample-count and three fixed quantile lines.
        add_metric_estimate(&mut total, 32, endpoint_labels)?;
        if total > MAX_RESPONSE_BODY_BYTES {
            return Some(total);
        }
    }
    Some(total)
}

fn checked_sum(values: &[usize]) -> Option<usize> {
    values
        .iter()
        .try_fold(0_usize, |total, value| total.checked_add(*value))
}

fn escaped_upper_bound(input_bytes: usize) -> Option<usize> {
    input_bytes.checked_mul(2)
}

fn add_metric_estimate(total: &mut usize, lines: usize, label_bytes: usize) -> Option<()> {
    let per_line = METRIC_LINE_UPPER_BOUND.checked_add(label_bytes)?;
    *total = total.checked_add(lines.checked_mul(per_line)?)?;
    Some(())
}

fn text_response(status: u16, body: &str, content_type: &'static str) -> HttpResponse {
    text_response_owned(status, body.as_bytes().to_vec(), content_type)
}

fn text_response_owned(status: u16, body: Vec<u8>, content_type: &'static str) -> HttpResponse {
    if body.len() > MAX_RESPONSE_BODY_BYTES {
        return response_too_large();
    }
    HttpResponse {
        status,
        content_type,
        body,
        cache_control: "no-cache",
        headers: Vec::new(),
    }
}

fn response_too_large() -> HttpResponse {
    fixed_json_error(503, "response exceeds server limit")
}

fn fixed_json_error(status: u16, message: &str) -> HttpResponse {
    let body = format!("{{\"error\":{message:?}}}\n").into_bytes();
    HttpResponse {
        status,
        content_type: "application/json; charset=utf-8",
        body,
        cache_control: "no-store",
        headers: Vec::new(),
    }
}

fn write_response(stream: &mut TcpStream, response: &HttpResponse) -> io::Result<()> {
    let mut head = String::with_capacity(384);
    let _ = write!(
        head,
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: {}\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n",
        response.status,
        reason_phrase(response.status),
        response.content_type,
        response.body.len(),
        response.cache_control,
    );
    for (name, value) in &response.headers {
        let _ = write!(head, "{name}: {value}\r\n");
    }
    head.push_str("\r\n");
    write_parts_with_timeout(stream, &[head.as_bytes(), &response.body], WRITE_TIMEOUT)
}

trait ResponseWrite: Write {
    fn set_response_timeout(&self, timeout: Duration) -> io::Result<()>;
}

impl ResponseWrite for TcpStream {
    fn set_response_timeout(&self, timeout: Duration) -> io::Result<()> {
        self.set_write_timeout(Some(timeout))
    }
}

fn write_parts_with_timeout(
    writer: &mut impl ResponseWrite,
    parts: &[&[u8]],
    timeout: Duration,
) -> io::Result<()> {
    let deadline = Instant::now() + timeout;
    for part in parts {
        write_before_deadline(writer, part, deadline)?;
    }
    let remaining = deadline
        .checked_duration_since(Instant::now())
        .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "response deadline exceeded"))?;
    writer.set_response_timeout(remaining)?;
    writer.flush()
}

fn write_before_deadline(
    writer: &mut impl ResponseWrite,
    mut bytes: &[u8],
    deadline: Instant,
) -> io::Result<()> {
    while !bytes.is_empty() {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| io::Error::new(io::ErrorKind::TimedOut, "response deadline exceeded"))?;
        writer.set_response_timeout(remaining)?;
        match writer.write(bytes) {
            Ok(0) => return Err(io::Error::from(io::ErrorKind::WriteZero)),
            Ok(written) => bytes = &bytes[written..],
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn reason_phrase(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        414 => "URI Too Long",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        _ => "Unknown",
    }
}

#[derive(Debug)]
struct LimitedWriter {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}

impl LimitedWriter {
    fn new(limit: usize) -> Self {
        Self {
            bytes: Vec::with_capacity(limit.min(8 * 1024)),
            limit,
            exceeded: false,
        }
    }

    fn exceeded(&self) -> bool {
        self.exceeded
    }

    fn into_inner(self) -> Vec<u8> {
        self.bytes
    }
}

impl Write for LimitedWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if buffer.len() > self.limit.saturating_sub(self.bytes.len()) {
            self.exceeded = true;
            return Err(io::Error::other("response size limit exceeded"));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_accepts_a_bounded_get_request() {
        let request = parse_request(
            b"GET /metrics?debug=true HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret-token\r\n\r\n",
        )
        .unwrap();
        assert_eq!(request.method, "GET");
        assert_eq!(request.route, "/metrics");
        assert_eq!(
            request.authorization,
            HeaderValue::Single("Bearer secret-token".to_owned())
        );
    }

    #[test]
    fn parser_enforces_request_line_and_header_byte_limits() {
        let long_target = "x".repeat(MAX_REQUEST_LINE_BYTES);
        let request = format!("GET /{long_target} HTTP/1.1\r\n\r\n");
        assert_eq!(
            parse_request(request.as_bytes()),
            Err(RequestError::RequestLineTooLong)
        );

        let long_value = "x".repeat(MAX_HEADER_BYTES);
        let request = format!("GET / HTTP/1.1\r\nX-Large: {long_value}\r\n\r\n");
        assert_eq!(
            parse_request(request.as_bytes()),
            Err(RequestError::HeadersTooLarge)
        );
    }

    #[test]
    fn parser_enforces_header_count_and_rejects_duplicate_authorization() {
        let mut request = String::from("GET /metrics HTTP/1.1\r\n");
        for index in 0..=MAX_HEADER_COUNT {
            let _ = write!(request, "X-{index}: value\r\n");
        }
        request.push_str("\r\n");
        assert_eq!(
            parse_request(request.as_bytes()),
            Err(RequestError::TooManyHeaders)
        );

        let request = parse_request(
            b"GET /metrics HTTP/1.1\r\nHost: localhost:9400\r\nAuthorization: Bearer secret-token\r\nAuthorization: Bearer secret-token\r\n\r\n",
        )
        .unwrap();
        assert!(!authorized(&request, "secret-token"));
    }

    #[test]
    fn request_read_timeout_is_classified_without_waiting() {
        struct TimeoutReader {
            sent_partial_header: bool,
        }

        impl Read for TimeoutReader {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                if !self.sent_partial_header {
                    self.sent_partial_header = true;
                    let partial = b"GET /metrics HTTP/1.1\r\nHost: local";
                    buffer[..partial.len()].copy_from_slice(partial);
                    return Ok(partial.len());
                }
                Err(io::Error::new(io::ErrorKind::TimedOut, "test timeout"))
            }
        }

        impl RequestRead for TimeoutReader {
            fn set_request_timeout(&self, _timeout: Duration) -> io::Result<()> {
                Ok(())
            }
        }

        assert_eq!(
            read_request(&mut TimeoutReader {
                sent_partial_header: false,
            }),
            Err(RequestError::Timeout)
        );
    }

    #[test]
    fn connection_queue_has_a_fixed_saturation_boundary() {
        assert!(!queue_is_full(CONNECTION_QUEUE_CAPACITY - 1));
        assert!(queue_is_full(CONNECTION_QUEUE_CAPACITY));
        assert!(queue_is_full(CONNECTION_QUEUE_CAPACITY + 1));
    }

    #[test]
    fn admission_limits_queued_and_active_connections_per_ip() {
        let admission = Arc::new(Admission::new());
        let peer = "127.0.0.1".parse().unwrap();
        let first = admission.try_acquire(peer).unwrap();
        let second = admission.try_acquire(peer).unwrap();
        assert!(admission.try_acquire(peer).is_none());
        drop(first);
        assert!(admission.try_acquire(peer).is_some());
        drop(second);
    }

    #[test]
    fn unauthenticated_loopback_requires_a_matching_host_authority() {
        assert_eq!(
            parse_request(b"GET /api/v1/report HTTP/1.1\r\n\r\n"),
            Err(RequestError::BadRequest)
        );
        assert_eq!(
            parse_request(
                b"GET /api/v1/report HTTP/1.1\r\nHost: localhost:9400\r\nHost: localhost:9400\r\n\r\n"
            ),
            Err(RequestError::BadRequest)
        );
        assert_eq!(
            parse_request(b"GET /api/v1/report HTTP/1.1\r\nHost:\r\n\r\n"),
            Err(RequestError::BadRequest)
        );

        let bound = "127.0.0.1:9400".parse().unwrap();
        let local =
            parse_request(b"GET /api/v1/report HTTP/1.1\r\nHost: localhost:9400\r\n\r\n").unwrap();
        let rebound =
            parse_request(b"GET /api/v1/report HTTP/1.1\r\nHost: attacker.example:9400\r\n\r\n")
                .unwrap();
        let legacy = parse_request(b"GET /api/v1/report HTTP/1.0\r\n\r\n").unwrap();
        assert!(loopback_host_authorized(&local.host, bound));
        assert!(!loopback_host_authorized(&rebound.host, bound));
        assert!(!loopback_host_authorized(&legacy.host, bound));
    }

    #[test]
    fn api_can_require_a_bearer_token() {
        assert!(bearer_authorized(
            Some("Bearer secret-token"),
            "secret-token"
        ));
        assert!(bearer_authorized(
            Some("bearer secret-token"),
            "secret-token"
        ));
        assert!(!bearer_authorized(None, "secret-token"));
        assert!(!bearer_authorized(
            Some("Bearer wrong-token"),
            "secret-token"
        ));
        assert!(!bearer_authorized(
            Some("Bearer  secret-token"),
            "secret-token"
        ));
        assert!(public_probe_route("/livez"));
        assert!(public_probe_route("/healthz"));
        assert!(!public_probe_route("/metrics"));
        assert!(!public_probe_route("/api/v1/report"));
    }

    #[test]
    fn routes_preserve_public_probes_and_protect_report_data() {
        let state = RwLock::new(State::default());
        let bound = "127.0.0.1:9400".parse().unwrap();
        let liveness =
            parse_request(b"GET /livez HTTP/1.1\r\nHost: localhost:9400\r\n\r\n").unwrap();
        assert_eq!(
            route_request(
                &liveness,
                &state,
                Duration::from_secs(10),
                Some("secret-token"),
                bound,
            )
            .status,
            200
        );

        let report =
            parse_request(b"GET /api/v1/report HTTP/1.1\r\nHost: localhost:9400\r\n\r\n").unwrap();
        assert_eq!(
            route_request(
                &report,
                &state,
                Duration::from_secs(10),
                Some("secret-token"),
                bound,
            )
            .status,
            401
        );

        let rebound =
            parse_request(b"GET /api/v1/report HTTP/1.1\r\nHost: attacker.example:9400\r\n\r\n")
                .unwrap();
        assert_eq!(
            route_request(&rebound, &state, Duration::from_secs(10), None, bound,).status,
            400
        );
    }

    #[test]
    fn response_writer_enforces_the_body_limit() {
        let oversized = text_response_owned(
            200,
            vec![b'x'; MAX_RESPONSE_BODY_BYTES + 1],
            "text/plain; charset=utf-8",
        );
        assert_eq!(oversized.status, 503);
        assert!(oversized.body.len() < MAX_RESPONSE_BODY_BYTES);
    }

    #[test]
    fn metrics_estimator_rejects_repeated_large_labels_before_encoding() {
        let mut report = Report::default();
        let mut gpu = crate::domain::Gpu {
            name: "x".repeat(16 * 1024),
            uuid: "y".repeat(16 * 1024),
            ..crate::domain::Gpu::default()
        };
        gpu.processes = vec![crate::domain::GpuProcess::default(); 100];
        report.gpus.push(gpu);
        assert!(prometheus_size_upper_bound(&report).unwrap() > MAX_RESPONSE_BODY_BYTES);
        assert_eq!(metrics_response(Some(&report), None).status, 503);
    }

    #[test]
    fn response_writes_use_one_absolute_deadline() {
        struct SlowProgressWriter {
            written: usize,
        }

        impl Write for SlowProgressWriter {
            fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
                std::thread::sleep(Duration::from_millis(3));
                self.written += 1;
                Ok(1)
            }

            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        impl ResponseWrite for SlowProgressWriter {
            fn set_response_timeout(&self, _timeout: Duration) -> io::Result<()> {
                Ok(())
            }
        }

        let mut writer = SlowProgressWriter { written: 0 };
        let error =
            write_parts_with_timeout(&mut writer, &[b"slow response"], Duration::from_millis(1))
                .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert_eq!(writer.written, 1);
    }

    #[test]
    fn public_exporter_constructor_enforces_remote_authentication() {
        let error = Exporter::start("127.0.0.1:0", Duration::from_secs(1), None).unwrap_err();
        assert!(error.to_string().contains("authentication is required"));

        let error = Exporter::start_unauthenticated_loopback("0.0.0.0:0", Duration::from_secs(1))
            .unwrap_err();
        assert!(error.to_string().contains("must use a loopback"));

        let error = Exporter::start(
            "127.0.0.1:0",
            Duration::from_secs(1),
            Some(" invalid\n".to_owned()),
        )
        .unwrap_err();
        assert!(error.to_string().contains("bearer token"));
    }
}
