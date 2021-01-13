import numpy as np


def complete(data):
    max_size = len(max(data, key=len))
    for i in len(data):
        data[i]


def fill_by_last_val(a):
    lens = np.array([len(item) for item in a.values()])
    ncols = lens.max()
    print(lens)
    last_ele = np.array([a_i[-1] for a_i in a.values()])
    out = np.repeat(last_ele[:, None], ncols, axis=1)
    mask = lens[:, None] > np.arange(lens.max())
    out[mask] = np.concatenate(list(a.values()))
    out = {k: v for k, v in enumerate(out)}
    print(out)
    exit(0)
    return out


if __name__ == '__main__':
    listo = {
        0: [{'val_loss': 0.761614978313446, 'val_acc': 0.8203125}, {'val_loss': 0.7444866299629211, 'val_acc': 0.8125},
            {'val_loss': 0.7416172623634338, 'val_acc': 0.7734375}],
        1: [{'val_loss': 0.6044541001319885, 'val_acc': 0.8203125},
            {'val_loss': 0.5943567156791687, 'val_acc': 0.83203125},
            {'val_loss': 0.5910449624061584, 'val_acc': 0.82421875},
            {'val_loss': 0.5897805094718933, 'val_acc': 0.8203125},
            {'val_loss': 0.5761075615882874, 'val_acc': 0.83203125},
            {'val_loss': 0.5684536099433899, 'val_acc': 0.8515625},
            {'val_loss': 0.5703654289245605, 'val_acc': 0.859375}],
        2: [{'val_loss': 0.5219595432281494, 'val_acc': 0.8515625},
            {'val_loss': 0.5271502137184143, 'val_acc': 0.8515625},
            {'val_loss': 0.5297966003417969, 'val_acc': 0.8515625}],
        3: [{'val_loss': 2.282625675201416, 'val_acc': 0.18359375}],
        4: [{'val_loss': 1.9691526889801025, 'val_acc': 0.43359375},
            {'val_loss': 1.9460170269012451, 'val_acc': 0.453125}, {'val_loss': 1.925710678100586, 'val_acc': 0.46875}],
        5: [{'val_loss': 0.45128509402275085, 'val_acc': 0.86328125},
            {'val_loss': 0.4476938247680664, 'val_acc': 0.8671875}, {'val_loss': 0.4388052523136139, 'val_acc': 0.875},
            {'val_loss': 0.43317461013793945, 'val_acc': 0.875}],
        6: [{'val_loss': 1.918778657913208, 'val_acc': 0.453125},
            {'val_loss': 1.8985230922698975, 'val_acc': 0.47265625}],
        7: [{'val_loss': 1.4024814367294312, 'val_acc': 0.515625},
            {'val_loss': 1.3647816181182861, 'val_acc': 0.5390625},
            {'val_loss': 1.3284419775009155, 'val_acc': 0.56640625},
            {'val_loss': 1.301019310951233, 'val_acc': 0.58203125}],
        8: [{'val_loss': 0.839232861995697, 'val_acc': 0.75}, {'val_loss': 0.8194305896759033, 'val_acc': 0.765625}],
        9: [{'val_loss': 0.48215943574905396, 'val_acc': 0.875}]}

    a = fill_by_last_val(listo)
    print(a)
