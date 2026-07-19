//! Thin executable adapter for the GPU Watchman library.

fn main() -> std::process::ExitCode {
    gpu_watchman::application::entrypoint()
}
