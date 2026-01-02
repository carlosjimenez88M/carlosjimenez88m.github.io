---
title: "MLops into Raspberry Pi 5"
date: 2025-02-23
draft: false
description: "A robust implementation of facilities for MLOps development"
categories: ["Edge Computing", "MLOps"]
tags: ["raspberry-pi", "mlflow", "edge-ai", "docker", "kubernetes", "vscode"]
---


One of the tools I use most for practicing **MLOps**, both for designing pipelines and APIs (for inference), is the **Raspberry Pi**. Today, I spent several hours trying to install **Visual Studio Code** to complement my **iPad Pro** as a development tool.

## **Why this setup?** 🤔

- Improve programming skills—I am a big fan of using **Weights & Biases (W&B)** to monitor the resource usage of each service I create.
- Using the **Raspberry Pi as a server** allows me to test **Edge computing** deployments.
- For **scalable prototype development**, it’s a great way to test artifacts and the **lifecycle of models**.
- When designing a model from **hyperparameters**, it helps me fine-tune **grid search** or **Bayesian methods** efficiently to optimize experimentation.
- Running **MLflow on Edge computing** enables **optimization** in model registry and updates.
- **Using Docker and Kubernetes** helps ensure **clean code** before committing changes.

There are many more reasons, but these are the main ones. Now, how do you set up **Raspberry Pi** to unlock its full power for MLOps?

---

## **🔧 Setting Up Raspberry Pi for MLOps**
First, install the **Raspberry Pi OS**. There are many tutorials, but I prefer the **official documentation**:

🔗 [Raspberry Pi OS Installation](https://www.raspberrypi.com/software/)

Next, **find the Raspberry Pi’s IP address** to connect to it from the iPad or another computer:

```bash
hostname -I
```

This will return something like `192.168.1.100 2601:123456`. You can then connect via SSH:

```bash
ssh pi@192.168.1.100
```

After entering the password (set during installation), **welcome to your new server**! 🎉

---

## **1️⃣ Installing Conda on Raspberry Pi**
By default, **Python** comes pre-installed. Now, install **Conda** using:

```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh
```

---

## **2️⃣ Installing MLflow**
Once Conda is installed, install **MLflow**:

```bash
pip install mlflow
```

---

## **3️⃣ Installing Docker**
To set up **Docker**, use:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io
```

For **advanced edge computing**, you can explore **DockerPi**:
🔗 [DockerPi](https://github.com/lukechilds/dockerpi)

---

## **4️⃣ Installing Tmux and Neovim**
To improve terminal workflow and coding experience, install **Tmux** and **Neovim**:

```bash
sudo apt update && sudo apt install -y tmux
```

For **Neovim**:

```bash
sudo apt update
sudo apt install -y neovim python3-neovim
```

### **🔹 Neovim Configuration for Python Development**
To configure **Neovim** for a better coding experience:

```bash
mkdir -p ~/.config/nvim
nano ~/.config/nvim/init.vim
```

Paste the following **Neovim config** for **line numbers, syntax highlighting, and autocompletion**:

```vim
" --- General Settings ---
set number            " Show line numbers
set mouse=a           " Enable mouse support
set cursorline        " Highlight the current line
set expandtab         " Use spaces instead of tabs
set shiftwidth=4      " Indentation size
set tabstop=4         " Tab size
set autoindent        " Maintain indentation
set smartindent       " Smart indentation
set background=dark   " Dark theme
set termguicolors     " Enable true colors
set encoding=utf-8    " UTF-8 support
syntax on             " Enable syntax highlighting
filetype plugin indent on " Enable plugins and indentation

" --- Plugins ---
call plug#begin('~/.vim/plugged')

" File explorer
Plug 'preservim/nerdtree'

" Autocompletion
Plug 'neoclide/coc.nvim', {'branch': 'release'}

" Language server support
Plug 'neovim/nvim-lspconfig'

" Improved syntax highlighting
Plug 'nvim-treesitter/nvim-treesitter', {'do': ':TSUpdate'}

" Status bar
Plug 'vim-airline/vim-airline'

call plug#end()

" --- Keybindings ---
nnoremap <C-n> :NERDTreeToggle<CR> " Open/close file explorer
nnoremap <C-p> :Files<CR> " Quick file search

" --- coc.nvim Configuration ---
let g:coc_global_extensions = ['coc-python', 'coc-json', 'coc-html', 'coc-tsserver']

" Auto-format Python code on save
autocmd BufWritePre *.py :Black
```

---

## **5️⃣ Installing and Configuring VS Code (`code-server`)**
### **🚀 Installing `code-server`**
```bash
curl -fsSL https://code-server.dev/install.sh | sh
```

### **🚀 Enabling `code-server`**
```bash
sudo systemctl enable --now code-server@$USER
```

If the service is masked:
```bash
sudo systemctl unmask code-server@$USER
sudo systemctl restart code-server@$USER
```

### **🚀 Checking `code-server` status**
```bash
sudo systemctl status code-server@$USER
```

If **active (running)** appears, `code-server` is working.

---

## **6️⃣ Configuring `code-server` for iPad Access**
Edit the config file:

```bash
nano ~/.config/code-server/config.yaml
```

Set the following configuration:

```yaml
bind-addr: 0.0.0.0:8080  # Change port if needed
auth: none               # Disable authentication (or use password)
cert: false              # No HTTPS
```

Save (`Ctrl + X`, `Y`, `Enter`), then restart:

```bash
sudo systemctl restart code-server@$USER
```

Check the correct port:

```bash
sudo netstat -tulpn | grep LISTEN
```

---

## **7️⃣ Accessing VS Code from the iPad**
Open **Safari or Chrome** on the iPad and enter:

```
http://<RASPBERRY_PI_IP>:8080
```

For example:

```
http://192.179.1.100:8080
```

To find your Raspberry Pi’s IP:

```bash
hostname -I
```

✅ If everything is correct, **VS Code (`code-server`) will open in the browser**.

---

## **8️⃣ Troubleshooting**
### **🔹 If `code-server` doesn’t load**
1. Check if `code-server` is running:
    ```bash
    sudo systemctl status code-server@$USER
    ```
2. Try accessing from Raspberry Pi itself:
    ```bash
    curl -v http://127.0.0.1:8080
    ```
3. Ensure Raspberry Pi and iPad are on the **same network**.

### **🔹 If the port is occupied (`EADDRINUSE`)**
1. Kill previous processes:
    ```bash
    pkill -f code-server
    ```
2. Restart `code-server`:
    ```bash
    sudo systemctl restart code-server@$USER
    ```
3. Verify the port:
    ```bash
    sudo netstat -tulpn | grep LISTEN
    ```


---

With this setup, your **Raspberry Pi 5** becomes a **powerful MLOps workstation**. 🚀🔥  
Let me know in the comments if you have questions!  
