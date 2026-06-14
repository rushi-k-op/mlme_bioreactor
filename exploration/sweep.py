import csv, sys
sys.path.insert(0, ".")
from basic_client import BioreactorClient, USER, PASSWORD

client = BioreactorClient()
client.login(USER, PASSWORD)
print("Logged in successfully.")

default = {"T": 35.0, "pH": 7.0, "F1": 2.0, "F2": 2.0, "F3": 2.0}

T_values   = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
pH_values  = [3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0, 8.5, 9.5]
F1_values  = [0.0, 0.5, 1.0, 1.5, 2.0]
F2_values  = [0.0, 0.5, 1.0, 1.5, 2.0]
F3_values  = [0.0, 0.5, 1.0, 1.5, 2.0]

with open("data/experiments.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "scale", "T", "pH", "F1", "F2", "F3", "Y", "cost_eur", "created_at"])

    # Temperature varying, other paramaters constant
    for T in T_values:
        for repeat in range(3):
            result = client.run("micro", T, default["pH"], default["F1"], default["F2"], default["F3"])
            writer.writerow([result["id"], "micro", T, default["pH"], default["F1"], default["F2"], default["F3"], result["Y"], result["cost_eur"], result["created_at"]])
            print(f"T={T}, repeat={repeat+1}, Y={result['Y']:.4f}")

    # pH varying, other paramaters constnt
    for pH in pH_values:
        for repeat in range(3):
            result = client.run("micro", default["T"], pH, default["F1"], default["F2"], default["F3"])
            writer.writerow([result["id"], "micro", default["T"], pH, default["F1"], default["F2"], default["F3"], result["Y"], result["cost_eur"], result["created_at"]])
            print(f"pH={pH}, repeat={repeat+1}, Y={result['Y']:.4f}")

    # F1 varying, other paramaters constant
    for F1 in F1_values:
        for repeat in range(3):
            result = client.run("micro", default["T"], default["pH"], F1, default["F2"], default["F3"])
            writer.writerow([result["id"], "micro", default["T"], default["pH"], F1, default["F2"], default["F3"], result["Y"], result["cost_eur"], result["created_at"]])
            print(f"F1={F1}, repeat={repeat+1}, Y={result['Y']:.4f}")

    # F2 varying, other paramaters constant
    for F2 in F2_values:
        for repeat in range(3):
            result = client.run("micro", default["T"], default["pH"], default["F1"], F2, default["F3"])
            writer.writerow([result["id"], "micro", default["T"], default["pH"], default["F1"], F2, default["F3"], result["Y"], result["cost_eur"], result["created_at"]])
            print(f"F2={F2}, repeat={repeat+1}, Y={result['Y']:.4f}")

    # F3 varying, other paramaters constant
    for F3 in F3_values:
        for repeat in range(3):
            result = client.run("micro", default["T"], default["pH"], default["F1"], default["F2"], F3)
            writer.writerow([result["id"], "micro", default["T"], default["pH"], default["F1"], default["F2"], F3, result["Y"], result["cost_eur"], result["created_at"]])
            print(f"F3={F3}, repeat={repeat+1}, Y={result['Y']:.4f}")

print("All sweeps done!")