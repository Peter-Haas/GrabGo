import collections
import time
from typing import Optional
from functools import reduce

from weight_utils import get_item_weights_from_stock_data

# from wiiboard import Wiiboard, BoardEvent
from door import Door
from item_filter import ItemFilter

from settings import (
    DEBUG,
    WEIGHT_SAMPLES,
    WEIGHT_BASE,
    DOOR_OPEN,
    DOOR_CLOSED,
    DOOR_LOCKED,
    WEIGHT_TOLERANCE,
    stock_data,
)


def sendMsg(data, *kwargs):
    print(">>>> sending: {data}".format(data=data))


class EventProcessor:
    def __init__(
        self,
        stock_data: list = stock_data,
        door: Door = None,
        board=None,  # Wiiboard
        weight_samples: int = WEIGHT_SAMPLES,
        averaged_total_weight: Optional[float] = 0,
        calibration_delta: Optional[float] = 0,
        data_smallest_present_weight: Optional[float] = 0.01,
        data_smallest_weight: Optional[float] = 0.01,
        data_item_weight: Optional[float] = 0.01,
        data_present_item_weight: Optional[float] = 0.01,
    ):
        self.board = board
        self.door = door
        self._doorStatus = (
            DOOR_CLOSED
        )  # gives last recorded door status. True is Open, False is closed
        self._doorLockStatus = False  # --> (
        # 	door.get_door_lock_status()
        # )  # gives last recorded door lock status. True is Open, False is closed
        self._measured = False
        self.done = False
        self.weight_samples = weight_samples
        self._measureCnt = 0
        self._events = list(range(weight_samples))
        self._takeMeasurement = False
        self.averaged_total_weight = (
            averaged_total_weight
        )  # weight the machine is loaded at. Get this value after restocking calibration
        self.averaged_item_weight = self.averaged_total_weight - WEIGHT_BASE
        self.calibration_delta = calibration_delta
        self.stock_data = stock_data
        self.data_smallest_present_weight = data_smallest_present_weight
        self.data_smallest_weight = data_smallest_weight
        self.data_item_weight = data_item_weight
        self.data_present_item_weight = data_present_item_weight

    def process_readings(self, boardevent):
        # Door is open, start taking measurements
        # weight in kg

        new_total_weight = boardevent.totalWeight
        if (
            self._takeMeasurement == True
            and boardevent.totalWeight > self.data_smallest_present_weight
        ):

            # First, we average out the new_total_weights over WEIGHT_SAMPLE number of samples
            self._events[self._measureCnt] = new_total_weight
            self._measureCnt += 1
            if self._measureCnt == self.weight_samples:
                self._measureCnt = 0
                self._sum = 0
                for x in range(0, self.weight_samples - 1):
                    self._sum += self._events[x]
                new_total_weight = self._sum / self.weight_samples

                # Detect changes only bigger than smallest weight
                current_weight_difference = self.averaged_total_weight - new_total_weight
                if abs(current_weight_difference) >= (
                    self.data_smallest_present_weight - WEIGHT_TOLERANCE
                ):

                    # Get only the weight of the items
                    new_item_weight = new_total_weight - WEIGHT_BASE

                    # if weight has changed (after averaging) do something
                    self.board.weight_log(
                        "current_weight_difference: {current_weight_difference}".format(
                            current_weight_difference=current_weight_difference
                        )
                    )
                    # if (
                    #     current_weight_difference > WEIGHT_TOLERANCE
                    #     or current_weight_difference < -WEIGHT_TOLERANCE
                    # ):
                    if current_weight_difference > 0:
                        # weight reduced, items removed
                        # find out combinations of items that might be purchases

                        # Note: set present_only/absent_only once we start updating the
                        # stock_data's 'present' value

                        event_type = "removed"
                        bagged_list = self.get_item_combinations_from_weight(
                            stock_data=self.stock_data,
                            current_weight_difference=current_weight_difference,
                            calibration_delta=self.calibration_delta,
                            # present_only=True,
                        )
                    else:
                        # weight increased, items added
                        # find out already missing items and figure out what combination of items from that list placed back
                        event_type = "put back"
                        bagged_list = self.get_item_combinations_from_weight(
                            stock_data=self.stock_data,
                            current_weight_difference=current_weight_difference,
                            calibration_delta=self.calibration_delta,
                            # absent_only=True,
                        )

                        self.board.weight_log(
                            "Items {event_type}: {bagged_list}".format(
                                event_type=event_type, bagged_list=bagged_list
                            )
                        )
                    self.update_current_weights(new_total_weight)
                    print("Measurement epoch complete.")

                    # self._doorStatus = self.door.get_door_status()

    def update_current_weights(self, averaged_total_weight: float):
        # Update the current weights to the new ones
        self.averaged_total_weight = averaged_total_weight
        self.averaged_item_weight = self.averaged_total_weight - WEIGHT_BASE
        data_item_weight, data_present_item_weight, data_smallest_weight, data_smallest_present_weight = get_item_weights_from_stock_data(
            data=self.stock_data
        )
        self.data_smallest_present_weight = data_smallest_present_weight
        self.data_smallest_weight = data_smallest_weight
        self.data_item_weight = data_item_weight
        self.data_present_item_weight = data_present_item_weight

    def get_item_combinations_from_weight(
        self,
        stock_data: list,
        current_weight_difference: float,
        calibration_delta: float,
        present_only: bool = False,
        absent_only: bool = False,
    ):
        # First prepare stock data by filtering out present/absent items
        current_weight_difference = abs(current_weight_difference)
        # present_or_absent_only_weight = self.data_item_weight
        if present_only and not absent_only:
            stock_data = [stock for stock in stock_data if stock["present"]]
        elif absent_only and not present_only:
            stock_data = [stock for stock in stock_data if not stock["present"]]

        # present_or_absent_only_weight = reduce(
        #     lambda a, b: a + b, [stock["weight"] for stock in stock_data]
        # )

        item_filter = ItemFilter(stock_data, current_weight_difference, calibration_delta)
        return item_filter.get_item_combinations(tolerance=WEIGHT_TOLERANCE)

    def get_item_weight_avg_for_calibration(self, boardevent):
        # We average out the new_total_weights over self.weight_samples number of samples
        self._events[self._measureCnt] = boardevent.totalWeight
        self._measureCnt += 1
        if self._measureCnt == self.weight_samples:
            self._measureCnt = 0
            self._sum = 0
            for x in range(0, self.weight_samples - 1):
                self._sum += self._events[x]
            new_averaged_total_weight = self._sum / self.weight_samples
            return new_averaged_total_weight

    @property
    def weight(self):
        if not self._events:
            return 0
        histogram = collections.Counter(round(num, 1) for num in self._events)
        return histogram.most_common(1)[0][0]

    def start_main_runloop(self):
        if self.door.get_door_status() == DOOR_OPEN and self._doorStatus == DOOR_CLOSED:  # --> and
            # self.door.get_door_lock_status()
            # == DOOR_UNLOCKED
            print("Door Opened")
            self._doorStatus = DOOR_OPEN
            time.sleep(2)
            self._measureCnt = 0
            while self.board.status == "Connected" and not self.done:

                # check if door closed
                if self.door.get_door_status() == DOOR_CLOSED and self._doorStatus == DOOR_OPEN:
                    self._doorStatus = DOOR_CLOSED
                    self.perform_post_door_closed()
                    break

                data = self.board.receivesocket.recv(25)
                boardevent = self.board.process_board_data(data)
                if boardevent:
                    self._takeMeasurement = True
                    self.process_readings(boardevent)
            return

    def perform_post_door_closed(self):
        print("Door Closed")
        self._takeMeasurement = False
        # --> self.door.set_door_lock_status(DOOR_LOCKED)


class StockCalibrator:
    """
	variable name key:
	data_: derived from stock_data.json
	averaged_: sampled over few samples
	present_: currently present in the machine
	total_: includes weight of the machine (BASE_WEIGHT)
	item_: just items

	"""

    def __init__(self, stock_data: list, door=None, board=None):
        self.door = door
        self.board = board
        self.stock_data = stock_data

    def calibrate_stock(self):
        self.board.weight_log("Calibrating stock...")
        # First, get weights from stock data
        data_item_weight, data_present_item_weight, data_smallest_weight, data_smallest_present_weight = get_item_weights_from_stock_data(
            data=self.stock_data
        )
        if DEBUG:
            calibration_delta = 0
            return (
                data_item_weight,
                data_present_item_weight,
                data_smallest_weight,
                data_smallest_present_weight,
                calibration_delta,
            )

            # Then, fill up machine with stock and calibrate for 300 epochs
        if (
            self.door.get_door_status()
            == DOOR_CLOSED
            # --> and self.door.get_door_lock_status() == DOOR_LOCKED
        ):
            # get board sensor readings
            averaged_total_weight = None
            processor = EventProcessor(door=self.door, board=self.board, weight_samples=300)
            while True:
                data = self.board.receivesocket.recv(25)
                # print("++++ receiving during calibration {}".format(data))
                boardevent = self.board.process_board_data(data)
                if boardevent:
                    averaged_total_weight = processor.get_item_weight_avg_for_calibration(
                        boardevent
                    )
                    # averaged_item_weight = averaged_total_weight - WEIGHT_BASE
                    # if averaged_total_weight:
                    #     calibration_delta = data_item_weight - averaged_item_weight
                if averaged_total_weight:
                    return (
                        data_item_weight,
                        data_present_item_weight,
                        data_smallest_weight,
                        data_smallest_present_weight,
                        averaged_total_weight,
                    )
        else:
            self.board.weight_log("Please close door and lock to begin calibration.")
            return (0, 0, 0, 0, 0)


class WeightSensor:
    def __init__(self, stock_data: list, door: Door, init_weights=None):
        self.stock_data = stock_data
        self.door = door
        self.init_weights = init_weights

    def weight_log(self, log_message):
        print(f"[INFO] Load Sensor: {log_message}")

    def initialize(self):
        """
        Abstract method for initialization.
        """

        raise NotImplementedError

    def perform_initial_calibration(self):
        """
        Abstract method for calibration.
        """

        raise NotImplementedError

    def get_current_weight(self):
        """
        Abstract method.
        """

        raise NotImplementedError

    def run_weight_event_loop(self):

        raise NotImplementedError
