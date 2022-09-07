# Various AI examples and lessons

## Prepare Python 3.8

On Ubuntu 22.04 LTS default 3.* branch of Python is 3.10. But there are not all library bindings ready for 3.10 at the time of writing (e.g. there is no tflite), so we use more supported 3.8. But there are no apt packages for Python 3.8 anymore in standard repositories of jessy. So install it from an [external one](https://www.linuxcapable.com/how-to-install-python-3-8-on-ubuntu-22-04-lts/):

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.8 python3.8-dev python3.8-venv python3.8-distutils python3.8-lib2to3 python3.8-gdbm python3.8-tk -y
```

Prepare venv:

```bash
python3.8 -m venv .venv

# on Linux
source .venv/bin/activate

# on Windows
.venv\Scripts\activate

pip install -r requirements.txt
```
