# ==============================================================================
# Mivi-v4 Task Runner (Justfile)
# Configured for low-memory, low-spec laptop environments (max 3 concurrent jobs)
# ==============================================================================

# Restrict Cargo to 3 parallel compiler worker threads to save RAM
export CARGO_BUILD_JOBS := "3"
export RUST_BACKTRACE := "1"

# Default recipe shows help
default:
    @just --list

# ------------------------------------------------------------------------------
# Build & Check Recipes (Low Resource Mode)
# ------------------------------------------------------------------------------

# Fast workspace syntax and type check without generating binaries
check:
    cargo check --workspace --jobs 3

# Compile workspace in debug mode (max 3 concurrent jobs)
build:
    cargo build --workspace --jobs 3

# Compile optimized release binary (max 3 concurrent jobs)
build-release:
    cargo build --workspace --release --jobs 3

# ------------------------------------------------------------------------------
# Testing & Quality Assurance
# ------------------------------------------------------------------------------

# Run full test suite with low memory concurrency (2 test threads)
test:
    cargo test --workspace --jobs 3 -- --test-threads=2

# Run clippy linter on all targets with zero tolerance for warnings
clippy:
    cargo clippy --workspace --all-targets --jobs 3 -- -D warnings

# Check code formatting without applying modifications
fmt-check:
    cargo fmt --all -- --check

# Auto-format all code in workspace
fmt:
    cargo fmt --all

# Run complete local verification pipeline (format check, clippy, tests)
verify: fmt-check clippy test

# ------------------------------------------------------------------------------
# Server & Runtime Runners
# ------------------------------------------------------------------------------

# Launch the OpenAI-compatible HTTP inference & Agent OS server
# Usage: just serve [model_path] [port] [host] [max_memory] [warn_memory]
serve model="models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf" port="8080" host="0.0.0.0" max_memory="3000" warn_memory="2400":
    cargo run --release --jobs 3 -- serve --model {{model}} --host {{host}} --port {{port}} --max-memory {{max_memory}} --warn-memory {{warn_memory}}

# Start interactive CLI terminal chat session with the model
# Usage: just chat [model_path] [temp]
chat model="models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf" temp="0.2":
    cargo run --release --jobs 3 -- chat --model {{model}} --temp {{temp}}

# Run hardware, SIMD, and system environment doctor diagnostics
doctor:
    cargo run --release --jobs 3 -- doctor

# Inspect GGUF model metadata, hyper-parameters, and tensor layouts
# Usage: just info <model_path>
info model="models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf":
    cargo run --release --jobs 3 -- info --model {{model}}

# Benchmark SIMD matrix-vector compute kernels on this machine
bench:
    cargo run --release --jobs 3 -- bench

# Generate the synthetic test GGUF model fixture (models/mivi-tiny-test.gguf)
generate-fixture:
    python3 training/export/generate_fixture.py

# List all persistent on-disk .kvc prefix cache files
cache-list:
    cargo run --release --jobs 3 -- cache list

# Clear all persistent on-disk .kvc prefix cache files
cache-clear:
    cargo run --release --jobs 3 -- cache clear

# ------------------------------------------------------------------------------
# Maintenance & Cleanup
# ------------------------------------------------------------------------------

# Clean build artifacts to reclaim disk space
clean:
    cargo clean
