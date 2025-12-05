# 🚗⛽ Gas Station Optimization Problem

This project explores the classic **gas station refueling optimization problem** — the road-trip scenario where you must travel from point A to point B with varying gas prices at different stations and a car with limited tank capacity.

The goal? **Minimize the total cost of fuel while reaching your destination.**

---

## Problem Overview

Imagine driving across multiple cities.  
Each city has a gas station with different prices, and your tank size is limited.  
So the key question becomes:

**Where should you refuel to minimize cost?**

This project compares two approaches to solve this problem.

---

## Approaches Compared

### 1️⃣ Naive (Brute Force) Approach  
- Tries many possible refueling combinations  
- Very slow for larger inputs  
- **Time Complexity:** `O(Δn³)`  
- Good for understanding, bad for scaling 🐌

### 2️⃣ Optimized Algorithm  
Based on the research paper **"To Fill or Not to Fill"**, using two strategic ideas:

1. **Fill up completely** if the next station is more expensive  
2. **Buy just enough fuel** if a cheaper station is ahead  

- Much faster  
- **Time Complexity:** `O(Δn² log n)`  
- Efficient even with many stations 🚀

---

## What's Inside

- **`optimized.py`** — Implements the fast `O(Δn² log n)` algorithm  
- **`brutevsoptimized.py`** — Compares naive vs optimized performance  
- **`*.png`** — Plots showing runtime differences and scalability  

---

## Running the Project

### Install dependencies:
```
pip install matplotlib numpy
```

### Run the optimized algorithm:
```
python3 optimized.py
python3 brutevsoptimized.py
```
