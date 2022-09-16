package main

import (
	"fmt"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

func checkErr(ret nvml.Return, operation string) bool {
	if ret != nvml.SUCCESS {
		fmt.Printf("%s failed: %v\n", operation, nvml.ErrorString(ret))
		return true
	}
	return false
}

func prop(ident string, ret nvml.Return, vals ...any) {
	if !checkErr(ret, ident) {
		fmt.Printf("%s: ", ident)
		fmt.Println(vals...)
	}
}

var knownArchitectures = map[nvml.DeviceArchitecture]string{
	nvml.DEVICE_ARCH_KEPLER:  "kepler",
	nvml.DEVICE_ARCH_MAXWELL: "maxwell",
	nvml.DEVICE_ARCH_PASCAL:  "pascal",
	nvml.DEVICE_ARCH_VOLTA:   "volta",
	nvml.DEVICE_ARCH_TURING:  "turing",
	nvml.DEVICE_ARCH_AMPERE:  "ampere",
}

func main() {
	ret := nvml.Init()
	if checkErr(ret, "Initialize NVML") {
		return
	}

	defer func() {
		ret := nvml.Shutdown()
		checkErr(ret, "Shutdown")
	}()

	count, ret := nvml.DeviceGetCount()
	if checkErr(ret, "DeviceGetCount") {
		return
	}

	nvmlVersion, r := nvml.SystemGetNVMLVersion()
	prop("SystemGetNVMLVersion", r, nvmlVersion)

	driverVersion, r := nvml.SystemGetDriverVersion()
	prop("SystemGetDriverVersion", r, driverVersion)

	cudaDriverVersion, r := nvml.SystemGetCudaDriverVersion()
	prop("SystemGetCudaDriverVersion", r, cudaDriverVersion)

	cudaDriverVersion2, r := nvml.SystemGetCudaDriverVersion_v2()
	prop("SystemGetCudaDriverVersion_v2", r, cudaDriverVersion2)

	for i := 0; i < count; i++ {
		fmt.Printf("\nDevice %d:\n", i)

		d, r := nvml.DeviceGetHandleByIndex(i)
		if checkErr(r, "DeviceGetHandleByIndex") {
			continue
		}

		uuid, r := d.GetUUID()
		prop("GetUUID", r, uuid)

		name, r := d.GetName()
		prop("GetName", r, name)

		mi, r := d.GetMemoryInfo()
		prop("GetMemoryInfo", r, mi)

		// Exists in nvidia driver 515
		// Doesn't exists in driver 470
		// mi2, r := d.GetMemoryInfo_v2()
		// prop("GetMemoryInfo_v2", r, mi2)

		// Exists in nvidia driver 515
		// Doesn't exists in driver 470
		// gpuCores, r := d.GetNumGpuCores()
		// prop("GetNumGpuCores", r, gpuCores)

		sn, r := d.GetSerial()
		prop("GetSerial", r, sn)

		attrs, r := d.GetAttributes()
		prop("GetAttributes", r, attrs)

		arch, r := d.GetArchitecture()
		prop("GetArchitecture", r, arch, knownArchitectures[arch])

		dm1, dm2, r := d.GetDriverModel()
		prop("GetDriverModel", r, dm1, dm2)

		da, r := d.GetDisplayActive()
		prop("GetDisplayActive", r, da)

		dm, r := d.GetDisplayMode()
		prop("GetDisplayMode", r, dm)

		fs, r := d.GetFanSpeed()
		prop("GetFanSpeed", r, fs)

		mds, r := d.GetPgpuMetadataString()
		prop("GetPgpuMetadataString", r, mds)

		br, r := d.GetBrand()
		prop("GetBrand", r, br)

		// Exists in nvidia driver 515
		// Doesn't exists in driver 470
		// bt, r := d.GetBusType()
		// prop("GetBusType", r, bt)

		cc1, cc2, r := d.GetCudaComputeCapability()
		prop("GetCudaComputeCapability", r, cc1, cc2)
	}
}
