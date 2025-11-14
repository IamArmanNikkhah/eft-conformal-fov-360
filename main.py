from src.simulator import Simulator

def main():
    user_id = input("Enter user ID (e.g., 2): ").strip()
    sim = Simulator(user_id, "avtrack360")
    log_df = sim.run()
    print("\nLog:")
    print(log_df)

if __name__ == "__main__":
    main()
