# Context Graph - Build & Maintenance Makefile
#
# All targets auto-trim stale deps to prevent unbounded disk growth.
# The 11-crate workspace produces ~600MB binaries per build; without
# trimming, target/ grows to 240GB+ within a few weeks.
#
# IMPORTANT: Building with Metal on Mac requires cmake parallelism limit
# to avoid OOM (llama-cpp-sys builds llama.cpp from source).
# The build targets below automatically set PATH to use the cmake wrapper.
#
# Key fixes for Metal builds:
# - Removed "llm" from default features in context-graph-mcp/Cargo.toml
# - cone_check FFI module gated behind #[cfg(feature = "cuda")]
# - cuda feature enables CUDA-specific code, metal feature uses Candle Metal
#
# Usage:
#   make build          Build release (MCP server + CLI)
#   make build-metal   Build with Metal support (Apple Silicon) [REQUIRES cmake wrapper]
#   make build-cuda    Build with CUDA support (NVIDIA)
#   make test           Run all workspace tests
#   make test-e2e       Run E2E hook tests only
#   make check          Quick workspace check (no codegen)
#   make check-metal    Quick workspace check with Metal [REQUIRES cmake wrapper]
#   make clean          Remove target/debug entirely
#   make clean-all      Remove entire target/ directory
#   make disk-check     Report disk usage and cleanable space
#   make trim           Trim stale deps only (no build)

# CMake wrapper path - limits parallelism to avoid OOM on Mac
CMAKE_WRAPPER := /tmp/cmake

# Check if cmake wrapper exists
HAS_CMAKE_WRAPPER := $(shell if [ -x "$(CMAKE_WRAPPER)" ]; then echo "1"; else echo "0"; fi)

# Build command with cmake wrapper if available
# The wrapper limits CMAKE_BUILD_PARALLEL_LEVEL to avoid OOM on Mac
ifdef HAS_CMAKE_WRAPPER
  ifeq ($(HAS_CMAKE_WRAPPER),1)
    CARGO_BUILD_CMD = PATH="/tmp:$(PATH)" cargo
  else
    CARGO_BUILD_CMD = cargo
  endif
else
  CARGO_BUILD_CMD = cargo
endif

.PHONY: build build-metal build-cuda test test-e2e test-metal check check-metal clean clean-all disk-check trim clippy setup-cmake

# --- Setup ---

# Create cmake wrapper to limit parallelism (required for llama-cpp-sys on Mac)
setup-cmake:
	@mkdir -p /tmp; \
	if [ ! -x "$(CMAKE_WRAPPER)" ]; then \
		echo "Creating cmake wrapper at $(CMAKE_WRAPPER)..."; \
		echo '#!/bin/bash' > $(CMAKE_WRAPPER); \
		echo 'export CMAKE_BUILD_PARALLEL_LEVEL=2' >> $(CMAKE_WRAPPER); \
		echo 'exec /opt/homebrew/bin/cmake "$$@"' >> $(CMAKE_WRAPPER); \
		chmod +x $(CMAKE_WRAPPER); \
		echo "Done. Use 'make build-metal' or 'make check-metal'"; \
	else \
		echo "cmake wrapper already exists at $(CMAKE_WRAPPER)"; \
	fi

# --- Build ---

build:
	cargo build --release
	@./scripts/trim-stale-deps.sh release

build-debug:
	cargo build
	@./scripts/trim-stale-deps.sh debug

# Metal build (Apple Silicon - Mac M1/M2/M3/M4)
# IMPORTANT: Run 'make setup-cmake' first to create the cmake wrapper
build-metal:
	@if [ ! -x "$(CMAKE_WRAPPER)" ]; then \
		echo "ERROR: cmake wrapper not found. Run 'make setup-cmake' first!"; \
		exit 1; \
	fi
	PATH="/tmp:$(PATH)" cargo build --release --no-default-features --features metal,llm --workspace
	@./scripts/trim-stale-deps.sh release

# CUDA build (NVIDIA RTX 5090)
build-cuda:
	cargo build --release --no-default-features --features cuda,llm --workspace
	@./scripts/trim-stale-deps.sh release

# --- Test ---

test:
	cargo test --workspace
	@./scripts/trim-stale-deps.sh debug

test-metal:
	@if [ ! -x "$(CMAKE_WRAPPER)" ]; then \
		echo "ERROR: cmake wrapper not found. Run 'make setup-cmake' first!"; \
		exit 1; \
	fi
	PATH="/tmp:$(PATH)" cargo test --no-default-features --features metal,llm --workspace
	@./scripts/trim-stale-deps.sh debug

test-e2e:
	cargo test -p context-graph-cli --test e2e
	@./scripts/trim-stale-deps.sh debug

test-mcp:
	cargo test -p context-graph-mcp
	@./scripts/trim-stale-deps.sh debug

# --- Check & Lint ---

check:
	cargo check --workspace --all-targets

check-metal:
	@if [ ! -x "$(CMAKE_WRAPPER)" ]; then \
		echo "ERROR: cmake wrapper not found. Run 'make setup-cmake' first!"; \
		exit 1; \
	fi
	PATH="/tmp:$(PATH)" cargo check --no-default-features --features metal,llm --workspace

check-cuda:
	cargo check --no-default-features --features cuda,llm --workspace

clippy:
	cargo clippy --workspace --all-targets -- -D warnings

# --- Cleanup ---

trim:
	@./scripts/trim-stale-deps.sh

clean:
	rm -rf target/debug
	@echo "Removed target/debug. Release binaries preserved."

clean-all:
	cargo clean
	@echo "Removed entire target/ directory."

clean-deep: clean-all
	./scripts/clean-build-artifacts.sh --aggressive

disk-check:
	@./scripts/clean-build-artifacts.sh --check
	@echo ""
	@./scripts/disk-guard.sh
