from tqdm import trange

def model_factory(model: str, config):


    return None


def create_tabular_gflow(shape, config):

    transforms = list()


    for ilayer in trange(config.num_layers):

        if config.layer_mg == "splinerq":
            transforms.append(
                SplineRQ(shape, num_bins=config.num_bins)
            )
        elif config.layer_mg == "mix":

        pass

    return None
