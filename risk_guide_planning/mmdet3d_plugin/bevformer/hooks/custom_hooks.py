from mmcv.runner.hooks.hook import HOOKS, Hook


@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters=every_n_inters

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            # runner.eval_model.load_state_dict(runner.model.state_dict())
            state_dict = runner.model.state_dict()
            # strip the module. prefix so it can be loaded into a bare model
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            runner.eval_model.load_state_dict(new_state_dict)
