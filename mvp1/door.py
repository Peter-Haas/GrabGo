import RPi.GPIO as io
from settings import (
    DEBUG,
    DOOR_PIN,
    DOOR_LOCK_PIN,
    DOOR_OPEN,
    DOOR_CLOSED,
    DOOR_LOCKED,
    DOOR_UNLOCKED,
)


class Door:
    def get_door_status(self) -> bool:
        if DEBUG:
            # emulating door was open and just being closed now
            return DOOR_OPEN
        else:
            return io.input(DOOR_PIN)

    def get_door_lock_status(self) -> bool:
        if DEBUG:
            return DOOR_UNLOCKED
        else:
            io.setup(DOOR_LOCK_PIN, io.IN)
            return io.input(DOOR_LOCK_PIN)

    def set_door_lock_status(self, status: bool):
        io.setup(DOOR_LOCK_PIN, io.OUT)
        io.output(DOOR_LOCK_PIN, status)
        print("Door Locked")
