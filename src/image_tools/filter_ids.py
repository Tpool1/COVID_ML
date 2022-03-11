import numpy as np

def filter_ids(array, clinical_ids):

    # list of array indices that need to be deleted
    del_indices = []

    # remove all non-digits from clinical_ids
    new_ids = []
    for id in clinical_ids:
        i = 0
        for char in id:
            if not char.isdigit():
                # remove char at i index
                first_part = id[:i]
                last_part = id[i+1:]
                id = first_part + last_part

            i = i + 1

        new_ids.append(id)

    clinical_ids = new_ids

    i = 0
    for img in array:
        id = img[-1]
        print("ID", id)
        print("clinical ids:", clinical_ids)
        if id not in clinical_ids:
            del_indices.append(i)

        i = i + 1

    array = np.delete(array, del_indices, axis=0)
    print("array shape after deletion:", array.shape)

    return array
