package main

import (
	"bytes"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
)

const (
	MAGIC_LIBC6   = "glibc-ld.so.cache"
	MAGIC_VERSION = "1.1"

	LD_ELF       = 0x0001
	LD_ARCH_MASK = 0xff00
)

var archNames = map[int]string{
	0x0000: "I386_LIB32",
	0x0300: "X8664_LIB64",
	0x0a00: "AARCH64_LIB64",
	0x0200: "FLAG_IA64_LIB64",
	0x0600: "FLAG_MIPS64_LIBN32",
	0x0700: "FLAG_MIPS64_LIBN64",
	0x0c00: "FLAG_MIPS_LIB32_NAN2008",
	0x0d00: "FLAG_MIPS64_LIBN32_NAN2008",
	0x0e00: "FLAG_MIPS64_LIBN64_NAN2008",
	0x0900: "FLAG_ARM_LIBHF",
	0x0b00: "FLAG_ARM_LIBSF",
}

/*
https://gist.github.com/asukakenji/f15ba7e588ac42795f421b48b8aede63
| GOARCH        | 32-bit | 64-bit |
| :------------ | :----: | :----: |
| `386`         | x      |        | LD_I386_LIB32     = 0x0000
| `amd64`       |        | x      | FLAG_X8664_LIB64  = 0x0300
| `amd64p32`    | x      |        | FLAG_X8664_LIBX32 = 0x0800
| `arm`         | x      |        |
| `arm64`       |        | x      | FLAG_AARCH64_LIB64 = 0x0a00
| `arm64be`     |        | x      |
| `armbe`       | x      |        |
| `loong64`     |        | x      |
| `mips`        | x      |        |
| `mips64`      |        | x      |
| `mips64le`    |        | x      |
| `mips64p32`   | x      |        |
| `mips64p32le` | x      |        |
| `mipsle`      | x      |        |
| `ppc`         | x      |        |
| `ppc64`       |        | x      | FLAG_POWERPC_LIB64 = 0x0500
| `ppc64le`     |        | x      |
| `riscv`       | x      |        |
| `riscv64`     |        | x      |
| `s390`        | x      |        |
| `s390x`       |        | x      | FLAG_S390_LIB64 = 0x0400
| `sparc`       | x      |        |
| `sparc64`     |        | x      | FLAG_SPARC_LIB64 = 0x0100
| `wasm`        |        | x      |
*/

// https://github.com/xmidt-org/libucresolv/blob/master/include/sysdeps/generic/dl-cache.h
// https://github.com/xmidt-org/libucresolv/blob/master/include/sysdeps/generic/ldconfig.h

type header_libc6 struct {
	Magic     [len(MAGIC_LIBC6)]byte
	Version   [len(MAGIC_VERSION)]byte
	NumLibs   uint32
	TableSize uint32
	Unused    [5]uint32
}

type entry_libc6 struct {
	Flags     int32
	Key       uint32
	Value     uint32
	OSVersion uint32
	Hwcap     uint64
}

func processCache(filename string, processEntry func(lib, path string, flags int)) {
	buf, err := ioutil.ReadFile(filename)
	if err != nil {
		log.Fatalln("failed to open", filename, err)
	}

	reader := bytes.NewReader(buf)

	hdr := header_libc6{}
	if err := binary.Read(reader, binary.LittleEndian, &hdr); err != nil {
		log.Fatalln("failed to decode header", err)
		return
	}

	cacheFmt := string(hdr.Magic[:])
	if cacheFmt != MAGIC_LIBC6 {
		log.Fatalf("unsupported ldcache format %s, only %s is supported\n", cacheFmt, MAGIC_LIBC6)
	}

	cacheVer := string(hdr.Version[:])
	if cacheVer != MAGIC_VERSION {
		log.Fatalf("unsupported ldcache format %s, only %s is supported\n", cacheVer, MAGIC_VERSION)
	}

	kvBuf := make([]byte, 4096)
	kvReader := bytes.NewReader(buf)
	kvRead := func(offset uint32) (string, error) {
		if _, err := kvReader.Seek(int64(offset), io.SeekStart); err != nil {
			log.Fatalln("failed to seek", err)
		}
		i := 0
		for {
			var b byte
			if b, err = kvReader.ReadByte(); err != nil {
				log.Fatalln("failed to read", err)
			}
			kvBuf[i] = b
			if b == 0 {
				break
			}
			i++
			if i == len(kvBuf) {
				log.Fatalln("buffer is too small")
			}
		}
		return string(kvBuf[:i]), nil
	}

	entry := entry_libc6{}
	for i := 0; i < int(hdr.NumLibs); i++ {
		if err := binary.Read(reader, binary.LittleEndian, &entry); err != nil {
			log.Fatalln("failed to read entry", i, err)
		}
		if entry.Flags&LD_ELF == 0 {
			continue
		}
		key, err := kvRead(entry.Key)
		if err != nil {
			log.Fatalln("failed to read key of entry", i, err)
		}
		value, err := kvRead(entry.Value)
		if err != nil {
			log.Fatalln("failed to read value of entry", i, err)
		}
		processEntry(key, value, int(entry.Flags))
	}
}

func getArchStr(flags int) string {
	arch := flags & LD_ARCH_MASK
	archName, ok := archNames[arch]
	if ok {
		archName = " (" + archName + ")"
	}
	return fmt.Sprintf("arch=0x%x%s", arch, archName)
}

func printLib(lib, path string, flags int) {
	fmt.Printf("%s = %s [flags=0x%x, %s]\n", lib, path, flags, getArchStr(flags))
}

func resolveLib(resolvingLib, cachedLib, libPath string, flags int) {
	if !strings.HasPrefix(cachedLib, resolvingLib) {
		return
	}

	fmt.Printf("\n%s [%s]\n", cachedLib, getArchStr(flags))

	for {
		fmt.Println("    -", libPath)

		fi, err := os.Lstat(libPath)
		if err != nil {
			fmt.Println("failed to check if symlink", libPath, err)
			break
		}

		if fi.Mode()&os.ModeSymlink == 0 {
			break
		}

		target, err := os.Readlink(libPath)
		if err != nil {
			fmt.Println("failed to resolve symlink", libPath, err)
			break
		}

		if filepath.IsAbs(target) {
			libPath = target
			continue
		}

		libPath = filepath.Join(filepath.Dir(libPath), target)
	}
}

const defaultCache = "/etc/ld.so.cache"

func main() {
	cacheFile := flag.String("i", "", "input cache file, default "+defaultCache)
	toResolve := flag.String("r", "", "library name to resolve")
	flag.Parse()

	if *cacheFile == "" {
		*cacheFile = defaultCache
	}

	processEntry := printLib
	if *toResolve != "" {
		processEntry = func(lib, path string, flags int) {
			resolveLib(*toResolve, lib, path, flags)
		}
	}

	processCache(*cacheFile, processEntry)
}
