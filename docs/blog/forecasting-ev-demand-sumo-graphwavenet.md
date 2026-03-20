# Forecasting EV Charging Demand with SUMO and Graph WaveNet: A Deep Dive

**Have you ever pulled into an EV charging station only to find every spot taken and a line of cars waiting?** As electric vehicle adoption accelerates, urban planners and grid operators are facing a monumental challenge: predicting *where* and *when* EV drivers will need to charge.

Traditional time-series forecasting often falls short because charging demand isn't just about time—it's highly dependent on the spatial layout of the city, road networks, and driver behavior. What if we could use microscopic traffic simulations and advanced graph neural networks to predict this demand before the bottleneck happens?

In this post, you'll learn:
- How to generate realistic EV charging datasets using the Eclipse SUMO traffic simulator
- The pitfalls of naively configuring EV battery physics in simulations
- How to extract spatiotemporal features from complex XML simulation outputs
- Why Graph WaveNet is a powerful architecture for this specific forecasting problem

Whether you're an AI researcher, a mobility engineer, or just curious about the intersection of smart grids and deep learning, this guide will walk you through a complete data engineering pipeline for EV demand forecasting.

---

## 1. The Challenge of Spatiotemporal Charging Data

To train an AI to predict charging demand, you need data. Lots of it. However, real-world charging station data is often heavily guarded by operators due to privacy concerns and competitive advantage. 

The solution? **Microscopic traffic simulation.** 

Using **SUMO (Simulation of Urban MObility)**, we can simulate thousands of individual vehicles navigating a city (like Berlin), draining their batteries, and making autonomous decisions to route to the nearest charging station when their State of Charge (SOC) drops below a certain threshold.

### Key Takeaway
- SUMO allows us to generate massive amounts of synthetic, yet physically grounded, spatiotemporal charging data to train predictive models without relying on restricted proprietary datasets.

---

## 2. Generating Realistic Simulation Scenarios

If you're going to simulate EVs, the physics have to make sense. A common mistake when configuring SUMO's `stationfinder` device is treating EV batteries like small remote-control cars rather than real vehicles.

Let's look at how to correctly configure an EV type in SUMO using Python.

### The Configuration Pitfall
Initially, we configured our simulated vehicles with an average battery capacity of `5,000 Wh` (5 kWh). For context, a modern EV like a Tesla Model 3 has around an `80,000 Wh` (80 kWh) battery. 

With a 5 kWh battery, our simulated cars were "charging" in about 80 seconds and only drawing 2 kWh per session. This resulted in zero wait times at stations, completely failing to simulate the peak-hour bottlenecks we wanted to study!

### The Fix: Realistic Parameter Grids
Here is how you properly set up EV physics in SUMO using Python's `xml.etree.ElementTree`:

```python
import xml.etree.ElementTree as ET

def generate_vtypes_xml(output_path, battery_mean, battery_std):
    """Generate SUMO vtypes.xml with realistic EV battery distributions."""
    root = ET.Element("additional")
    vtype = ET.SubElement(root, "vType", id="DEFAULT_VEHTYPE")
    
    # Enable the battery device
    ET.SubElement(vtype, "param", key="has.battery.device", value="true")
    
    # Define realistic capacity (e.g., mean 75,000 Wh)
    capacity_dist = f"normc({battery_mean},{battery_std},10000,150000)"
    ET.SubElement(vtype, "param", key="device.battery.capacity", value=capacity_dist)
    
    # TRICK: Initialize charge level to a low distribution (e.g., 25% SOC)
    # If everyone starts at 100%, nobody will charge during a 2-hour simulation!
    charge_mean = battery_mean * 0.25  
    charge_std = battery_mean * 0.15
    charge_dist = f"normc({charge_mean},{charge_std},1000,150000)"
    ET.SubElement(vtype, "param", key="device.battery.chargeLevel", value=charge_dist)

    tree = ET.ElementTree(root)
    tree.write(output_path)
```

**⚠️ Common Mistake:** Forgetting to lower the initial `chargeLevel`. If your simulation only runs for 2 hours (e.g., morning rush hour) and all EVs start at 100% battery, no one will ever visit a charging station. By initializing the fleet at around 25% SOC, we force realistic mid-commute charging behavior.

---

## 3. Parsing XML Output into Machine Learning Tensors

SUMO outputs rich, event-based XML files (`tripinfos.xml`, `charging_output.xml`, `stops_output.xml`). Deep learning models don't read XML; they read tensors. We need to aggregate these discrete events into fixed temporal bins (e.g., 5-minute intervals) for each spatial node (charging station).

Our goal is to create a Numpy array with the shape:
`(Num_Stations, Time_Bins, Features)` -> e.g., `(53, 24, 6)`

### Extracting the Ground Truth State of Charge (SOC)
One crucial feature for our model is the average SOC of vehicles when they arrive at the station. Calculating this requires linking the energy charged back to the specific vehicle's maximum capacity.

Here is how to extract that cleanly from SUMO's `charging_output.xml`:

```python
from lxml import etree

def parse_charging_stations(xml_path):
    stations = {}
    # Stream the XML to avoid loading massive files into memory
    for event, elem in etree.iterparse(str(xml_path), tag="chargingStation"):
        cs_id = elem.get("id")
        vehicles = []
        
        for veh in elem.findall("vehicle"):
            arrival_capacity = 0.0
            max_capacity = 35000.0 # Fallback
            
            # The exact arrival capacity is recorded in the first 'step' element
            first_step = veh.find("step")
            if first_step is not None:
                arrival_capacity = float(first_step.get("actualBatteryCapacity", "0"))
                max_capacity = float(first_step.get("maximumBatteryCapacity", "35000"))

            # Calculate actual SOC % at arrival
            arrival_soc = arrival_capacity / max_capacity if max_capacity > 0 else 0
            
            vehicles.append({
                "id": veh.get("id"),
                "energy_charged": float(veh.get("totalEnergyChargedIntoVehicle", "0")),
                "arrival_soc": arrival_soc
            })
            
        stations[cs_id] = vehicles
        elem.clear() # Free memory
        
    return stations
```

**Performance Consideration:** Notice the use of `etree.iterparse()` and `elem.clear()`. SUMO XML outputs for thousands of vehicles can quickly exceed several gigabytes. Iterative parsing prevents your memory from overflowing during the data engineering phase.

---

## 4. Why Graph WaveNet for Demand Forecasting?

Once we have our tensor `(Batch, Stations, Time, Features)`, how do we predict the next hour of demand? 

You could use an LSTM, but LSTMs only understand time. If Station A (downtown) gets full, drivers will redirect to Station B (suburbs). There is a *spatial* relationship dictated by the road network.

This is where **Graph WaveNet** shines.

### The Architecture
Graph WaveNet combines two powerful concepts:
1. **Dilated 1D Convolutions (WaveNet):** Captures long-range temporal dependencies much faster than an LSTM.
2. **Graph Convolutional Networks (GCN):** Captures spatial dependencies between stations based on distance.

```python
# A conceptual snippet of the forward pass
def forward(self, x, predefined_adj_matrix):
    # x shape: (Batch, Nodes, Time, Features)
    
    # 1. Temporal feature extraction
    temporal_out = self.dilated_temporal_conv(x)
    
    # 2. Adaptive Graph Convolution
    # Graph WaveNet learns hidden spatial relationships that aren't in the predefined matrix!
    adaptive_adj = F.softmax(F.relu(torch.mm(self.node_embeddings_1, self.node_embeddings_2.T)), dim=1)
    combined_adj = predefined_adj_matrix + adaptive_adj
    
    # 3. Spatial feature extraction
    spatial_out = self.graph_conv(temporal_out, combined_adj)
    
    return self.output_mlp(spatial_out)
```

### ✅ Best Practice: The Adaptive Adjacency Matrix
The "secret sauce" of Graph WaveNet is the **Adaptive Adjacency Matrix**. While we provide the model with a predefined matrix based on the physical distance between charging stations, Graph WaveNet learns its own node embeddings. By taking the dot product of these embeddings, the model discovers hidden relationships. 

For example, two stations might be geographically far apart, but connected by a fast highway, making them highly correlated in terms of overflow traffic. The adaptive matrix learns this automatically.

---

## 5. Troubleshooting the Pipeline

Building this pipeline isn't without its headaches. Here are common issues we encountered:

**Issue: "My model predicts 0 for all stations!"**
- **Cause**: Data sparsity. In a 2-hour simulation, many stations might never see a single EV. If 95% of your tensor is zeros, the model will simply learn to predict zero to minimize MSE.
- **Solution**: Filter out inactive stations before training, or use a custom loss function that heavily penalizes false negatives (missing a charging event).

**Issue: `avg_soc_arrival` is wildly incorrect (e.g., 95%).**
- **Cause**: Hardcoding constants in the parser. We initially divided the charged energy by a hardcoded `35000 Wh` instead of the specific vehicle's actual capacity, skewing the SOC metric.
- **Solution**: Always extract the exact `maximumBatteryCapacity` parameter directly from the simulation step logs, as shown in the code snippet above.

---

## 6. Conclusion

You've now seen how to bridge the gap between microscopic traffic simulation and deep learning. By configuring SUMO with realistic EV parameters, efficiently parsing massive XML logs into structured tensors, and applying a spatiotemporal model like Graph WaveNet, we can create robust forecasts for EV infrastructure.

As we continue to optimize our cities for electrification, these predictive models will be critical in deciding where to build the next charging hub and how to route vehicles during peak hours to avoid gridlock.

**Call to Action:**
Are you working on intelligent transportation systems? Try generating your own dataset using SUMO's `stationfinder` device, and see if a simple GCN can predict your traffic flows! Let me know what challenges you run into.

**Next Steps & Resources:**
- [Eclipse SUMO Documentation: Electric Vehicles](https://sumo.dlr.de/docs/Models/Electric.html)
- [Graph WaveNet Paper (Wu et al., 2019)](https://arxiv.org/abs/1906.00121)
- [LXML Iterparse Guide for Large Files](https://lxml.de/parsing.html#iterparse-and-iterwalk)
