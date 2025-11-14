from src.simulator import Simulator

def main():
    user_id = "2"
    dataset_name = "avtrack360"

    sim = Simulator(user_id, dataset_name)
    sim.run()

if __name__ == "__main__":
    main()
