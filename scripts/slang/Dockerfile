# syntax=docker/dockerfile:1.7
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl jq tar libxml2 \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    case "${TARGETARCH}" in \
        "amd64") slang_pkg="linux-x86_64" ;; \
        "arm64") slang_pkg="linux-aarch64" ;; \
        *) echo "Unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    api_json="$(curl -fsSL https://api.github.com/repos/shader-slang/slang/releases/latest)"; \
    asset_url="$(printf '%s' "${api_json}" | jq -r --arg suffix "${slang_pkg}.tar.gz" '.assets[] | select(.name | endswith($suffix)) | .browser_download_url' | head -n1)"; \
    test -n "${asset_url}"; \
    curl -fL "${asset_url}" -o /tmp/slang.tar.gz; \
    mkdir -p /opt/slang; \
    tar -xzf /tmp/slang.tar.gz -C /opt/slang --strip-components=1; \
    rm -f /tmp/slang.tar.gz; \
    ln -sf /opt/slang/slangc /usr/local/bin/slangc

WORKDIR /work
