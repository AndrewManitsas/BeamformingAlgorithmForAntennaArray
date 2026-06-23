---
config:
  layout: elk
  theme: redux
---
flowchart LR
 subgraph subGraph0["The Conductor"]
        Main["main.py"]
  end
 subgraph subGraph1["The Core Logic"]
        Opt["optimization.py"]
        Math["array_math.py"]
  end
 subgraph subGraph2["Validation & Output"]
        Metrics["metrics.py"]
        Vis["visualization.py"]
  end
    Config[\"config.csv"\] -- "1. Loads Parameters" --> Main
    Main -- "2. Passes Scenario Angles & SNR" --> Opt
    Opt <-- "3. Requests Weights & Patterns iteratively" --> Math
    Opt -- "4. Returns Optimized Weights & Dummies" --> Main
    Main -- "5. Passes Final Weights for Grading" --> Metrics
    Metrics -- "6. Returns Deviations, SLL, SINR" --> Main
    Main -- "7. Passes Data for Headless Plotting" --> Vis
    Vis -- "8. Saves .png file to disk" --> Plots[\"plots/ Directory"\]
    Main -- "9. Writes Min/Max/Mean/Std" --> TextLog[\"simulation_results.txt"\]

     Main
     Opt
     Math
     Metrics
     Vis
     Config
     Plots
     TextLog