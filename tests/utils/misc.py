def compare_lists(list1, list2):
    for item1, item2 in zip(list1, list2):
        if item1 != item2:
            return False
    return True
