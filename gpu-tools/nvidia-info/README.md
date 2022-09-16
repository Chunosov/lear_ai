# Read NVIDIA Info

Example of usage of the [NVML](https://docs.nvidia.com/deploy/nvml-api/) library and its [go bindings](https://github.com/NVIDIA/go-nvml).

```bash
$ make run

SystemGetNVMLVersion: 11.470.141.03
SystemGetDriverVersion: 470.141.03
SystemGetCudaDriverVersion: 11040
SystemGetCudaDriverVersion_v2: 11040

Device 0:
GetUUID: GPU-13ea4ce9-ae8f-a362-ea1b-c3815e7b72e0
GetName: NVIDIA GeForce MX150
GetMemoryInfo: {2099904512 2094661632 5242880}
GetSerial failed: Not Supported
GetAttributes failed: Not Supported
GetArchitecture: 4 pascal
GetDriverModel failed: Not Supported
GetDisplayActive: 0
GetDisplayMode: 0
GetFanSpeed failed: Not Supported
GetPgpuMetadataString failed: Not Supported
GetBrand: 5
GetCudaComputeCapability: 6 1
```