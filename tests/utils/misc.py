def compare_lists(list1, list2):
    for item1, item2 in zip(list1, list2):
        if item1 != item2:
            return False
    return True


def get_abbreviated_dict_string(d: dict) -> str:
    s = ""
    for key, value in d.items():
        str_value = str(value)
        if len(str_value) > 100:
            str_value = str_value[:100] + "..."
        s += f"{key}: {str_value}\n"
    return s
