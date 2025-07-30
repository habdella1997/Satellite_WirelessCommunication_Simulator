# Project:
This project was developed as part of my Master Thesis at Northeastern University in 2020. 
Results and findings were published (Best Paper Award) https://ieeexplore.ieee.org/abstract/document/9469460
Presentation can be found in MasterThesis_Hussam.ppt



# Satellite Wireless Communication Simulator

This repository contains a Python-based simulator for modeling and analyzing wireless satellite communication systems, including THz and other advanced spectral bands.
The project allows the following functionality: 

1. Loading an exisiting Satellite Constellation From an exisiting TLE Set (ex: Starlink).
2. Creating your own satellite constellation (focus is on the LEO orbit)
3. Dynamically update satellite coordinates (SSP point) with high percesision. Satellite tracking is done through the use of computing Kepler's equation, updating the orbital parameters, and generating the polar, GEC coordinates
   The rotational Coordinates, and then generating the LOOKUP Angles.
4. Defining the Channel Models via ITU recommendation for NTN communication: (ITU-R P.676)
5. Computing the SNR, BER, Data RATE
6. Find the Cross Link Route (shortest next hop) {Routing / NW Layer} and compute propagation Delay
---
To Use This Porject:
You need two API KEYS defined in constants.py
WEATHER_API_KEY = <INSERT KEY HERE>
GOOFLE_MAP_API  = <INSERT KEY HERE>
For the Wearther API key you can generate one from this link: https://api.openweathermap.org/data/2.5/weather
---

---

## 🚀 Getting Started

These instructions will get your environment set up and the simulator running locally.

---

### 🛠️ Prerequisites

You need either:

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- OR Python 3.8+ with `pip` and `virtualenv`

---

### ✅ Option 1: Setup with Conda (Recommended)

Create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate satcomsim
```

---

### ✅ Option 2: Setup with Pip

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # on Windows use: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Simulation

After setup, run the main simulation script:

```bash
python main.py
```

You can modify simulation parameters within `main.py`.

---

## 📌 Notes

- Some modules may rely on external data sources or TLE satellite input files.
- Visualization features may require `matplotlib`, `plotly`, or similar libraries.

---

## 🧑‍💻 Author

- Hussam Abdellatif

---

