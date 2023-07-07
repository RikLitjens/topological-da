def rank(func, item, full_list):
    # get stats about item
    item_func_val = func(item)

    # and the rest of the list
    full_list_func_val = sorted([func(x) for x in full_list])

    # count the occurences of the item func_val
    amount = full_list_func_val.count(item_func_val)

    # cannot happen
    if amount == 0:
        assert False

    idx = full_list_func_val.index(item_func_val)
    rank_sum = 0

    # 1-indexed
    for i in range(1, amount + 1):
        # 1/|X|
        rank_sum += (idx + i) / len(full_list)

    return rank_sum / amount


# more efficient when calculating a lot
def create_rank_dict(func, full_list):
    full_list_func_val = sorted([func(x) for x in full_list])

    dic = {}
    for item in full_list:
        item_func_val = func(item)
        item_amount = full_list_func_val.count(item_func_val)

        # cannot happen
        if item_amount == 0:
            assert False

        idx = full_list_func_val.index(item_func_val)
        rank_sum = 0

        # 1-indexed
        for i in range(1, item_amount + 1):
            # 1/|X|
            rank_sum += (idx + i) / len(full_list)

        dic[item] = rank_sum / item_amount

    return dic


def find_base_node(superpoints):
    sorted_superpoints = sorted(superpoints, key=lambda x: x[2])
    return sorted_superpoints[0]


if __name__ == "__main__":
    print(rank(lambda x: x, 3, [4, 5, 5, 3, 7]))
