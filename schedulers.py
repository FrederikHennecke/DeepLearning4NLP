from itertools import cycle


class Scheduler:
    def __init__(self, dataloaders, reset=True):
        self.dataloaders = dataloaders
        self.names = list(dataloaders.keys())
        if reset:
            self.reset()

    def reset(self):
        self.sst_iter = iter(self.dataloaders["sst"])
        self.para_iter = iter(self.dataloaders["para"])
        self.sts_iter = iter(self.dataloaders["sts"])
        self.steps = {"sst": 0, "para": 0, "sts": 0}

    def get_SST_batch(self):
        try:
            return next(self.sst_iter)
        except StopIteration:
            self.sst_iter = cycle(self.dataloaders["sst"])
            return next(self.sst_iter)

    def get_Paraphrase_batch(self):
        try:
            return next(self.para_iter)
        except StopIteration:
            self.para_iter = cycle(self.dataloaders["para"])
            return next(self.para_iter)

    def get_STS_batch(self):
        try:
            return next(self.sts_iter)
        except StopIteration:
            self.sts_iter = cycle(self.dataloaders["sts"])
            return next(self.sts_iter)

    def get_batch(self, name):
        if name == "sst":
            return self.get_SST_batch()
        elif name == "para":
            return self.get_Paraphrase_batch()
        elif name == "sts":
            return self.get_STS_batch()
        else:
            raise ValueError(f"Invalid batch name: {name}")

    def process_named_batch(self, objects_group, args, name, apply_optimization):
        batch = self.get_batch(name)
        process_fn, gradient_accumulations = None, 0
        if name == "sst":
            process_fn = process_sentiment_batch
            gradient_accumulations = args.gradient_accumulations_sst
        elif name == "para":
            process_fn = process_paraphrase_batch
            gradient_accumulations = args.gradient_accumulations_para
        elif name == "sts":
            process_fn = process_similarity_batch
            gradient_accumulations = args.gradient_accumulations_sts
        else:
            raise ValueError(f"Invalid batch name: {name}")

        loss_of_batch = 0
        for _ in range(gradient_accumulations):
            loss_of_batch += process_fn(batch, objects_group, args)

        self.steps[name] += 1
        if apply_optimization:
            step_optimizer(objects_group, args, step=self.steps[name])

        return loss_of_batch

class RandomScheduler(Scheduler):
    def __init__(self, dataloaders):
        super().__init__(dataloaders, reset = True)

    def process_one_batch(self, epoch,, num_epochs, objects_group, args):
        name = random.choice(self.names)
        return name, self.process_named_batch(objects_group, args, name)

class RoundRobinScheduler(Scheduler):
    def __init__(self, dataloaders):
        super().__init__(dataloaders, reset = False)
        self.reset()

    def reset(self):
        self.index = 0
        return super().reset()

    def process_one_batch(self, epoch, num_epochs, objects_group, args):
        name = self.names[self.index]
        self.index = (self.index + 1) % len(self.names)
        return name, self.process_named_batch(objects_group, args, name)

class PalScheduler(Scheduler):
    def __init__(self, dataloaders):
        super().__init__(dataloaders, reset = False)
        self.sizes = np.array([len(dataloaders[dataset]) for dataset in self.names])
        self.reset()

    def process_one_batch(self, epoch, num_epochs, objects_group, args, apply_optimization = True):
        alpha = 0.2
        if num_epochs > 1:
            alpha = 1 - 0.8 * (epoch - 1) / (num_epochs - 1)
        probs = self.sizes ** alpha
        probs /= np.sum(probs)
        name = np.random.choice(self.names, p=probs)
        return name, self.process_named_batch(objects_group, args, name, apply_optimization = apply_optimization)

    def process_several_batches_with_control(self, epoch, num_epochs, objects_group, args, num_batches):
        schedule = ['sst', 'para', 'sts']
        alpha = 0.2
        if num_epochs > 1:
            alpha = 1 - 0.8 * (epoch - 1) / (num_epochs - 1)
        probs = self.sizes ** alpha
        probs /= np.sum(probs)
        probs_biased = (probs * num_batches - 1) / (num_batches - 3)
        probs_biased = np.clip(probs_biased, 0.025, 1)
        probs_biased /= np.sum(probs_biased)
        schedule += np.random.choice(self.names, size = num_batches - 3, p=probs_biased).tolist()
        random.shuffle(schedule)

        losses = []
        for task in schedule:
            loss = self.process_named_batch(objects_group, args, task, apply_optimization = False)
            losses.append(loss)
        return schedule, losses
