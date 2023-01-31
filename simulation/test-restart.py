import random
import logging
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Damage, Electrics, Timer

def run_scenario(vehicle, bng):
    bng.scenario.restart()
    bng.control.pause()
    kph = 0.
    vehicle.sensors.poll()
    sensors = vehicle.sensors
    while kph < 30 and sensors["damage"]["damage"] <= 1.:
        steering_cmd = 0.
        vehicle.control(throttle=1., steering=steering_cmd, brake=0.)
        bng.control.step(1, wait=True)
        vehicle.sensors.poll()
        sensors = vehicle.sensors
        print(f"Zeroed steering_cmd={steering_cmd:3f}\t\tsteering_input={sensors['electrics']['steering_input']:3f}")
        kph = 3.6 * sensors['electrics']['wheelspeed']
        start_time = sensors['timer']['time']
    while sensors['timer']['time'] - start_time < 5 and sensors["damage"]["damage"] <= 1.:
        throttle = random.uniform(0.0, 1.0)
        steering = random.uniform(-1.0, 1.0)
        vehicle.control(throttle=throttle, steering=steering, brake=0.0)

        bng.control.step(1, wait=True)
        vehicle.sensors.poll()
        sensors = vehicle.sensors
        print(f"Random steering_cmd={steering:3f}\t\tsteering_input={sensors['electrics']['steering_input']:3f}")

def main():
    random.seed(1703)
    # bng = BeamNGpy("localhost", 64256, home='path/to/binary', user='path/to/instance')
    bng = BeamNGpy("localhost", 64256, home='F:/BeamNG.tech.v0.27.1.0', user='C:/Users/Meriel/Documents/BeamNG.tech')
    bng.open(launch=True)
    scenario = Scenario('automation_test_track', 'Evaluator test', description="Ensuring the evaluator can repair any fault")
    vehicle = Vehicle('ego_vehicle', model='etk800', licence='EGO', color="green")
    vehicle.attach_sensor('electrics', Electrics())
    vehicle.attach_sensor('damage', Damage())
    vehicle.attach_sensor('timer', Timer())
    spawn_point = {'pos': (174.92, -289.7, 120.7), 'rot_quat': (0.0, 0.0, 0.7115, 0.7027)}
    scenario.add_vehicle(vehicle, pos=spawn_point['pos'], rot_quat=spawn_point['rot_quat'])
    scenario.make(bng)
    bng.settings.set_deterministic(30)
    bng.scenario.load(scenario)
    bng.scenario.start()
    bng.control.pause()
    for i in range(5):
        run_scenario(vehicle, bng)
    bng.close()

if __name__ == '__main__':
    logging.getLogger('PIL').setLevel(logging.WARNING)
    main()