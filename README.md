# VPilot
Scripts and tools to easily communicate with [DeepGTAV](https://github.com/ai-tor/DeepGTAV). In the future a self-driving agent will be implemented.

<img src="http://forococheselectricos.com/wp-content/uploads/2016/07/tesla-autopilot-1.jpg" alt="Self-Driving Car" width="900px">

## How it works

VPilot uses JSON over TCP sockets to start, configure and send/receive commands to/from [DeepGTAV](https://github.com/ai-tor/DeepGTAV) by using the Python DeepGTAV libraries. 

_dataset_ and _drive_ serve as examples to collect a dataset using DeepGTAV and giving the control to an agent respectively. Minimum requirement: Python 3.2 or greater

### Examples

[SantosNet](https://github.com/cpgeier/SantosNet) - Simple model and tools to use with VPilot
