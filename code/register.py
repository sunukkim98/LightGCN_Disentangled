import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book', 'yelp-review', 'beauty', 'sports', 'toys']:
    if world.dataset == 'yelp-review':
        dataset = dataloader.Loader(path="../data 2/yelp/")
    elif world.dataset == 'beauty' or world.dataset == 'sports' or world.dataset == 'toys':
        dataset = dataloader.Loader(path="../data 2/"+world.dataset)
    else:
        dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    'dlgn': model.DLightGCN
}