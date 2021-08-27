import fire

from disco_split.processing.mine import from_mine
from disco_split.processing.transforms.complex import simples_to_complex
from disco_split.processing.transforms.tree import simples_to_tree, args_to_tree


class BuildFuncs(object):

    def s2c(self, samples, out_file=None, complex_col="complex", simp_cols=["sent1", "sent2"], seed=False):
        simples_to_complex(samples, out_file=out_file, complex_col=complex_col, simp_cols=simp_cols, seed=seed)
        return "Complete."

    def s2t(self, samples, out_file=None, keep_original=False, conn_col="connective"):
        simples_to_tree(samples, out_file, keep_original, conn_col=conn_col)
        return "Complete."

    def a2t(self, samples, out_file=None, arg_cols=["arg1", "arg2"], make_ys=True, bin_mask=None, oracle=False):
        args_to_tree(samples, out_file, arg_cols=arg_cols, make_ys=make_ys, bin_mask=bin_mask, oracle=oracle)
        return "Complete."
    
    def mine(self, mine_dir, out_file=None, item_type="simple", sample_limit=None, strict=False):
        from_mine(mine_dir, out_file=out_file, item_type=item_type, sample_limit=sample_limit, strict=strict)


if __name__ == '__main__':
    fire.Fire(BuildFuncs)
