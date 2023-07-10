from itertools import groupby
import json
from settings import WEIGHT_PRECISION, WEIGHT_TOLERANCE
from functools import reduce


class Knapsack:
    @classmethod
    def knapsack01_dp(self, stock_data, limit):
        # Bounded Knapsack problem
        items = sum(([(item, wt, val)] * n for item, wt, val, n in stock_data), [])
        table = [[0 for w in range(limit + 1)] for j in range(len(items) + 1)]

        for j in range(1, len(items) + 1):
            item, wt, val = items[j - 1]
            for w in range(1, limit + 1):
                if wt > w:
                    table[j][w] = table[j - 1][w]
                else:
                    table[j][w] = max(table[j - 1][w], table[j - 1][w - wt] + val)
        result = []
        w = limit
        for j in range(len(items), 0, -1):
            was_added = table[j][w] != table[j - 1][w]

            if was_added:
                item, wt, val = items[j - 1]
                result.append(items[j - 1])
                w -= wt

        return result


class ItemFilter:
    def __init__(self, stock_data: list, max_weight: float, calibration_delta: float):
        self.stock_data = stock_data
        self.processed_stock_data = self.process_for_knapsack(stock_data)
        self.current_weight_difference = max_weight
        self.max_weight = int(round(max_weight * WEIGHT_PRECISION, len(str(WEIGHT_PRECISION)) - 1))
        self.calibration_delta = int(
            round(calibration_delta * WEIGHT_PRECISION, len(str(WEIGHT_PRECISION)) - 1)
        )

    def process_for_knapsack(self, stock_data: list):
        # convert stock object format to knapsack format: [(id, weight, value, qty), (id, weight,
        # value, qty),... ]
        knapsack_stock_data = []
        for item in stock_data:
            weight = int(round(item["weight"] * WEIGHT_PRECISION, len(str(WEIGHT_PRECISION)) - 1))
            knapsack_stock_data.append(((item["id"], weight, 1, item["qty"])))
        return knapsack_stock_data

    def get_item_combinations(self, tolerance: float):
        # generate different max_weights to get more combinations
        item_combinations = []
        lower_max_weight = self.max_weight - self.calibration_delta
        if lower_max_weight < 0:
            lower_max_weight = 0
        upper_max_weight = self.max_weight + self.calibration_delta
        calibrated_max_weight = [lower_max_weight, self.max_weight, upper_max_weight]

        # Apply filtering algorithm for each max_weight
        for mw in calibrated_max_weight:
            if mw != 0:
                combination = Knapsack.knapsack01_dp(self.processed_stock_data, mw)
                item_combinations.append(combination)
                print("combination: {combination}".format(combination=combination))

                # return combinations as stock objects
        real_item_combinations = []
        for bagged in item_combinations:
            print(
                "Bagged the following %i items\n  " % len(bagged)
                + "\n  ".join(
                    "%i off: %s" % (len(list(grp)), item[0])
                    for item, grp in groupby(sorted(bagged))
                )
            )
            print(
                "for a total value of %i and a total weight of %i"
                % (sum(item[2] for item in bagged), sum(item[1] for item in bagged))
            )
            comb = []

            for item in bagged:
                matched_item = [
                    {"id": data["id"], "weight": data["weight"]}
                    for data in self.stock_data
                    if data["id"] == item[0]
                ]
                if matched_item:
                    comb.append(matched_item[0])
            real_item_combinations.append(comb)

        # return real_item_combinations
        return self.final_item_filter(
            real_item_combinations, tolerance, abs(self.current_weight_difference)
        )

    def final_item_filter(
        self, combinations: list, tolerance: float, current_weight_difference: float
    ):
        # removes combinations that are much less than what real values should be
        final_combinations = []
        for items in combinations:
            comb_list = []
            combined_weight = 0
            for item in items:
                combined_weight += item.get("weight", 0)
                comb_list.append(item)
            if combined_weight >= current_weight_difference - tolerance:
                final_combinations.append(comb_list)
        return final_combinations


if __name__ == "__main__":
    with open("./stock_data.json") as f:
        stock_data = json.loads(f.read())
    stock_data = stock_data["products"]
    current_weight_difference = 2.09
    calibration = 0.8
    tolerance = 0.2
    item_filter = ItemFilter(stock_data, current_weight_difference, calibration)
    bagged_list = item_filter.get_item_combinations(tolerance=tolerance)
    print("\nReal:")
    for comb in bagged_list:
        print("combination: {comb}".format(comb=comb))
        # print(json.dumps(bagged_list, sort_keys=True, indent=4))
