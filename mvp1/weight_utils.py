def get_item_weights_from_stock_data(data: list):
    data_item_weight = 0
    data_present_item_weight = 0
    data_smallest_weight = 0
    data_smallest_present_weight = 0
    if data:
        data_smallest_weight = data[0]["weight"]
        present_items = [item for item in data if item["present"] == True]
        first_present_item = present_items[0] if present_items else {}
        data_smallest_present_weight = first_present_item.get("weight", 0)
        for product in data:
            data_item_weight += product["weight"] * product["qty"]
            if product["present"] == True:
                data_present_item_weight += product["weight"] * product["qty"]
                if product["weight"] < data_smallest_present_weight:
                    data_smallest_present_weight = product["weight"]
            if product["weight"] < data_smallest_weight:
                data_smallest_weight = product["weight"]

    return (
        data_item_weight,
        data_present_item_weight,
        data_smallest_weight,
        data_smallest_present_weight,
    )
