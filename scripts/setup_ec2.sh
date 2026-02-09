#!/usr/bin/env bash
# ==========================================================================
# Tennis Virtual Ads — EC2 g4dn.xlarge Setup (Phase 1: Pre-Reboot)
#
# Usage:
#   ssh -A -i <key>.pem ubuntu@<ec2-host>
#   mkdir -p ~/capstone-repos && cd ~/capstone-repos
#   git clone git@github.com:enriquedlh97/tennis-virtual-ads.git
#   cd tennis-virtual-ads
#   bash scripts/setup_ec2.sh
#
# What this does:
#   1. Configures git (author name/email) + vim + terminal prompt
#   2. Installs system packages (OpenCV deps, build tools)
#   3. Installs NVIDIA driver 570 (T4 GPU)
#   4. Installs uv (Python package manager)
#   5. Clones sibling repos into ~/capstone-repos/
#   6. Reboots (required for NVIDIA driver)
#
# After reboot, reconnect and run: bash scripts/setup_ec2_post_reboot.sh
# ==========================================================================

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
GIT_USER_NAME="Enrique Diaz de Leon Hicks"
GIT_USER_EMAIL="enriquedlh97@hotmail.com"

NVIDIA_DRIVER_VERSION="570"

WORKSPACE_DIR="$HOME/capstone-repos"

SIBLING_REPOS=(
    "git@github.com:SebastianBitsch/tennis-court-tracker.git"
    "git@github.com:mmmmmm44/tennis_court_detection.git"
    "git@github.com:AggieSportsAnalytics/CourtCheck.git"
    "git@github.com:yastrebksv/TennisCourtDetector.git"
)

# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------
info() {
    echo ""
    echo "========================================"
    echo "  $1"
    echo "========================================"
    echo ""
}

# --------------------------------------------------------------------------
# 1. Git config
# --------------------------------------------------------------------------
info "1/6  Configuring Git"

git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"

echo "  user.name  = $(git config --global user.name)"
echo "  user.email = $(git config --global user.email)"

# --------------------------------------------------------------------------
# 2. Shell environment (vim + terminal prompt with git branch)
# --------------------------------------------------------------------------
info "2/6  Configuring shell (vim, terminal prompt)"

# Only append if not already configured (idempotent)
MARKER="# === tennis-virtual-ads shell config ==="

if ! grep -q "$MARKER" "$HOME/.bashrc"; then
    cat << 'BASHRC_EOF' >> "$HOME/.bashrc"

# === tennis-virtual-ads shell config ===

# Vim as default editor
export EDITOR=vim

# Terminal prompt with git branch in yellow
parse_git_branch() {
    git rev-parse --abbrev-ref HEAD 2>/dev/null | sed 's/\(.*\)/ (\x1b[33m\1\x1b[0m)/'
}
export PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]$(parse_git_branch)\$ '

# === end tennis-virtual-ads shell config ===
BASHRC_EOF
    echo "  Shell config appended to ~/.bashrc"
else
    echo "  Shell config already present in ~/.bashrc — skipping"
fi

# shellcheck source=/dev/null
source "$HOME/.bashrc" || true

# --------------------------------------------------------------------------
# 3. System packages (OpenCV deps + build tools)
# --------------------------------------------------------------------------
info "3/6  Installing system packages"

sudo apt update -y
sudo apt install -y \
    build-essential \
    libgl1 \
    libsm6 \
    libxrender1 \
    libglib2.0-0

echo "  System packages installed"

# --------------------------------------------------------------------------
# 4. NVIDIA driver
# --------------------------------------------------------------------------
info "4/6  Installing NVIDIA driver $NVIDIA_DRIVER_VERSION"

if command -v nvidia-smi &> /dev/null; then
    echo "  nvidia-smi already available — skipping driver install"
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    sudo apt install -y "nvidia-driver-${NVIDIA_DRIVER_VERSION}"
    echo "  NVIDIA driver $NVIDIA_DRIVER_VERSION installed (reboot required)"
fi

# --------------------------------------------------------------------------
# 5. uv (Python package manager)
# --------------------------------------------------------------------------
info "5/6  Installing uv"

if command -v uv &> /dev/null; then
    echo "  uv already installed: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"

    # Ensure it persists across sessions
    if ! grep -q 'cargo/env' "$HOME/.bashrc" && ! grep -q '.local/bin' "$HOME/.bashrc"; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
    fi

    echo "  uv installed: $(uv --version)"
fi

# --------------------------------------------------------------------------
# 6. Clone sibling repos
# --------------------------------------------------------------------------
info "6/6  Cloning sibling repos into $WORKSPACE_DIR"

for repo_url in "${SIBLING_REPOS[@]}"; do
    repo_name=$(basename "$repo_url" .git)
    target_dir="$WORKSPACE_DIR/$repo_name"

    if [ -d "$target_dir" ]; then
        echo "  $repo_name — already exists, skipping"
    else
        echo "  Cloning $repo_name ..."
        git clone "$repo_url" "$target_dir"
    fi
done

# --------------------------------------------------------------------------
# Summary + Reboot
# --------------------------------------------------------------------------
info "Phase 1 complete!"

echo "  Installed:"
echo "    - Git config: $GIT_USER_NAME <$GIT_USER_EMAIL>"
echo "    - Vim as default editor"
echo "    - Terminal prompt with git branch"
echo "    - System packages (OpenCV deps)"
echo "    - NVIDIA driver $NVIDIA_DRIVER_VERSION"
echo "    - uv: $(uv --version 2>/dev/null || echo 'installed')"
echo ""
echo "  Workspace: $WORKSPACE_DIR"
ls -1 "$WORKSPACE_DIR"
echo ""
echo "  ⚠️  Rebooting in 5 seconds (NVIDIA driver requires reboot)..."
echo "  After reboot, reconnect and run:"
echo ""
echo "    cd ~/capstone-repos/tennis-virtual-ads"
echo "    bash scripts/setup_ec2_post_reboot.sh"
echo ""

sleep 5
sudo reboot
