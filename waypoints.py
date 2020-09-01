#!/usr/bin/env python3

import asyncio
from mavsdk import (System, MissionItem)


class Waypoint:
  def __init__(self, latitude, longitude):
    self.latitude = latitude
    self.longitude = longitude

async def print_mission_progress(drone):
    async for mission_progress in drone.mission.mission_progress():
        if mission_progress.current_item_index != 0:
            print(f"Achieved Waypoint {mission_progress.current_item_index}. ")
            percent = (mission_progress.current_item_index/mission_progress.mission_count)*100
            print(f"Mission progress: {round(percent,2)}%")

async def print_position(drone):
    """ Prints the altitude when it changes """

    previous_position = None

    async for position in drone.telemetry.position():
        if position != previous_position:
            previous_position = position
            print(f"Achived {position}")


async def print_flight_mode(drone):
    """ Prints the flight mode when it changes """

    previous_flight_mode = None

    async for flight_mode in drone.telemetry.flight_mode():
        if flight_mode is not previous_flight_mode:
            previous_flight_mode = flight_mode
            print(f"Flight mode changed to {flight_mode}")


async def observe_is_in_air(drone, running_tasks):
    """ Monitors whether the drone is flying or not and
    returns after landing """

    was_in_air = False

    async for is_in_air in drone.telemetry.in_air():
        if is_in_air:
            was_in_air = is_in_air

        if was_in_air and not is_in_air:
            for task in running_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            await asyncio.get_event_loop().shutdown_asyncgens()
            return

async def run():
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"Drone discovered with UUID: {state.uuid}")
            break

    # Start parallel tasks
    print_flight_mode_task = asyncio.ensure_future(print_flight_mode(drone))

    running_tasks = [print_flight_mode_task]
    termination_task = asyncio.ensure_future(observe_is_in_air(drone, running_tasks))

    asyncio.ensure_future(print_mission_progress(drone))

    #asyncio.ensure_future(print_position(drone))

    waypoint1 = Waypoint(47.398039859999997, 8.5455725400000002)
    waypoint2 = Waypoint(47.398036222362471, 8.5450146439425509)
    waypoint3 = Waypoint(47.397825620791885, 8.5450092830163271)

    flight_altitude = 5
    flight_speed = 10

    mission_items = []
    mission_items.append(MissionItem(waypoint1.latitude,
                                     waypoint1.longitude,
                                     flight_altitude,
                                     flight_speed,
                                     True,
                                     float('nan'),
                                     float('nan'),
                                     MissionItem.CameraAction.NONE,
                                     float('nan'),
                                     float('nan')))

    mission_items.append(MissionItem(waypoint2.latitude,
                                     waypoint2.longitude,
                                     flight_altitude,
                                     flight_speed,
                                     True,
                                     float('nan'),
                                     float('nan'),
                                     MissionItem.CameraAction.NONE,
                                     float('nan'),
                                     float('nan')))

    mission_items.append(MissionItem(waypoint3.latitude,
                                     waypoint3.longitude,
                                     flight_altitude,
                                     flight_speed,
                                     True,
                                     float('nan'),
                                     float('nan'),
                                     MissionItem.CameraAction.NONE,
                                     float('nan'),
                                     float('nan')))

    await drone.mission.set_return_to_launch_after_mission(True)

    print("-- Uploading mission")
    await drone.mission.upload_mission(mission_items)

    print("-- Arming")
    await drone.action.arm()

    print("-- Mission")
    await drone.mission.start_mission()

    # Wait until the drone is landed (instead of exiting after 'land' is sent)
    await termination_task

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())