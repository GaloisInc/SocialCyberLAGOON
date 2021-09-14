"""Walt GCN

# 2021-08-31
Fixed a few issues; added MLP re-mix for additional power between graph hops.
Completely separated test set from both training and validation sets.
Also added cycling entity sets for more variety; before was fixed random seed.
(base naive classifications now presented as list of `val/test`)

Without data: [0.856/1.099 0.992/1.008 0.796/1.004 0.994/1.006] / [5.889/14.379 16.529/11.480 14.806/10.455 11.585/20.533]
With:         [0.821/0.832 0.814/0.843 0.522/0.772 0.931/1.087] / [6.643/12.164 8.017/6.820 17.242/4.905 11.843/11.614]
Type only, no bad words:
              [0.936/1.023 0.617/1.331 0.945/0.878 0.942/0.885] / [19.027/8.383 5.930/15.938 13.547/4.255 11.852/9.502]
2-hops: Overfits.

Summary: including data results in a 11.7% reduction in MSE.

# 2021-08-26
Testing with N = 3 trials over 10000 steps
Results reported as fraction of naive ; naive validation loss
Main model with hops=3 crashes due to OOM; subsampling required
Main model with hops=2 over-fits slightly; got one OOM. Focusing on hops=1 for now

OK, looking at with type / badword data vs without. All 1hop, 10k batches, 5k validation samples
With: [.96 .96 .98] / [15.65 15.70 15.49]
Without: [.99 .99 .98] / [15.97 15.61 15.33]

So, fairly consistent gain of about 3% predictive power. Just bad words count
and sociotechnical interactions. This is more promising than it sounds!
"""

from .data import BatchManager
from .model import Model

from lagoon.db.connection import get_session
import lagoon.db.schema as sch

import argparse
from pprint import pprint
import torch

def main():
    ap = argparse.ArgumentParser()
    Model.argparse_setup(ap)
    args = ap.parse_args()

    model = Model.argparse_create(args.__dict__)
    model.build()
    pprint(model.argparse_hparams())
    batch_size = model.config.batch_size
    batches_print = 100  # 50
    batches_test = 5000 // batch_size  # 5000 // batch_size

    # Fetching a batch is expensive! So do mini-batches of mini-batches, and
    # deliberately over-fit to each mini-batch
    batches_train_group = 50  # Max training batches to hold in memory;
                              # also size gathered to calculate naive guess
    batches_train_iters = 10  # Fetch a new training batch every this many iterations

    # Once built, push to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    batch_man = BatchManager(batch_size, model.config.embed_size,
            window_size=model.config.window_size,
            hops=model.config.hops,
            data_badwords=model.config.data_badwords,
            data_cheat=model.config.data_cheat,
            data_type=model.config.data_type,
            precache=10,
            device=device)

    print(f'Dealing with {batch_man.count_entity()} people')
    def losses(ins, outs, y):
        stats = [i[-1] for i in ins]
        loss_y = (outs - y).pow_(2)
        naive_y = torch.tensor([
                # Use global if no local guess
                s['naive_guess'] if 'naive_guess' in s else naive_guess
                for s in stats],
                dtype=float, device=device)[:, None]
        loss_naive = (outs - naive_y).pow_(2)
        loss_scale = torch.tensor([s.get('weight', 1.) for s in stats],
                device=device)[:, None]
        assert loss_y.shape == loss_naive.shape
        assert loss_y.shape == loss_scale.shape
        loss_y *= loss_scale
        loss_naive *= loss_scale
        return loss_y, loss_naive

    def calc_naive_guess(batches):
        num = 0.
        den = 0.
        for tb in batches:
            tb_out = tb[1].cpu()
            assert tb_out.shape == (len(tb[0]), 1), tb_out.shape
            for ti, t in enumerate(tb[0]):
                w = t[-1]['weight']
                den += w
                num += w * tb_out[ti, 0].item()
        return num / den

    def get_test_set(group_num, naive_guess=None):
        assert group_num in [1, 2]
        test_batches = [batch_man.fetch_batch(model.type_embeds, group_num)
                for _ in range(batches_test)]
        # Use a single naive guess across all test batches which is derived from
        # training data, rather than the default per-batch naive guess.
        if naive_guess is None:
            # Use average over these batches
            naive_avg = calc_naive_guess(test_batches)
        else:
            # Use exact value
            naive_avg = naive_guess
        for tb in test_batches:
            for t in tb[0]:
                t[-1]['naive_guess'] = naive_avg
        return test_batches

    # Pre-cache the entire training data pipeline and naive guess
    train_batches = [batch_man.fetch_batch(model.type_embeds, 0)
            for _ in range(batches_train_group)]
    naive_guess: float = calc_naive_guess(train_batches)

    # Use training data rather than naive guess optimal for each set; use `None`
    # for specific guesses.
    ng = naive_guess
    val_batches = get_test_set(1, naive_guess=ng)
    # I think we want to use a separate naive classifier for test_batches --
    # while it may skew our results toward the conservative side, it removes a
    # random variable which would increase the necessary number of trials.
    test_batches = get_test_set(2, naive_guess=ng)

    def save_model():
        """Saves out the model and parameters required to create it."""
        data = {'hparams': model.argparse_hparams(),
                'state_dict': model.state_dict(),
                # We should keep the ordered list of entities for the batch
                # manager around; this contains the train/test split information
                'entities': batch_man._entities,
        }
        torch.save(data, 'model.chkpt')

    def validate_losses(batches, debug_print=False):
        model.eval()
        with torch.no_grad():
            loss = 0.
            loss_naive = 0.  # Also look at predicting future == past as naive
            printed = 0
            printed_0 = 0
            for val_batch_i, (ins, outs) in enumerate(batches):
                # Overwrite new embeddings values
                ins = batch_man.prepare_batch_ins(model.type_embeds, ins)
                y = model(ins)
                y_loss, naive_loss = losses(ins, outs, y)
                loss += y_loss.mean()
                loss_naive += naive_loss.mean()

                if not debug_print:
                    continue

                for stats_i, [*_, stats] in enumerate(ins):
                    if (printed < 10 and stats['weight'] > 1
                            or printed_0 == 0 and stats['weight'] <= 1):
                        # Don't print large tensors
                        stats = {k: v for k, v in stats.items()
                                if not k.startswith('data_')}
                        print(f'{stats} -> {y[stats_i].item():.3f}; actual {outs[stats_i].item():.3f}')
                        printed += 1
                        if stats['weight'] <= 1:
                            printed_0 += 1
            loss /= len(batches)
            loss_naive /= len(batches)
        return torch.stack([loss, loss_naive]).cpu()

    def validate(i):
        nonlocal loss_avg, loss_naive_avg, loss_avg_div, loss_min, \
                loss_test_min, loss_test_naive
        print(f'step: {i}')

        v_loss, v_loss_naive = validate_losses(val_batches, debug_print=True)

        if v_loss.item() < loss_min:
            loss_min = v_loss
            save_model()
            # Run test set on this new network (if eval time took longer, we'd
            # want to duplicate model)
            loss_test_min, loss_test_naive = validate_losses(test_batches)

        print(f'loss_val: {v_loss.item()}')
        print(f'loss_val_min: {loss_min.item()}')
        print(f'loss_val_naive: {v_loss_naive.item()}')
        print(f'loss_test_min: {loss_test_min.item()}')
        print(f'loss_test_naive: {loss_test_naive.item()}')
        print(f'loss_train: {(loss_avg / loss_avg_div).item()}')
        print(f'loss_train_naive: {(loss_naive_avg / loss_avg_div).item()}')

        loss_avg.fill_(0.)
        loss_naive_avg.fill_(0.)
        loss_avg_div = 1e-30

    loss_avg = torch.tensor(0., device=device)
    loss_min = torch.tensor(1e30, device=device)
    loss_test_min = torch.tensor(0., device=device)
    loss_test_naive = torch.tensor(0., device=device)
    loss_naive_avg = torch.tensor(0., device=device)
    loss_avg_div = 1e-30
    for batch_num in range(model.config.train_epochs):
        if batch_num % batches_print == 0:
            validate(batch_num)

        # Data fetching
        if batch_num != 0 and batch_num % batches_train_iters == 0:
            while len(train_batches) >= batches_train_group:
                train_batches.pop(0)
            train_batches.append(batch_man.fetch_batch(model.type_embeds, 0))

        model.train()

        # Remix training rows
        b1 = torch.randint(len(train_batches), (batch_size,)).tolist()
        ins = []
        outs = []
        for bi, b in enumerate(b1):
            ins.append(train_batches[b][0][bi])
            outs.append(train_batches[b][1][bi])
        outs = torch.stack(outs)

        # Set up embedding assignment, which provides some grads // uses new
        # values
        ins = batch_man.prepare_batch_ins(model.type_embeds, ins)
        y = model(ins)
        loss_y, loss_naive = losses(ins, outs, y)
        loss = loss_y.mean()
        loss.backward()
        model.opt.step()
        model.opt.zero_grad()

        with torch.no_grad():
            loss_avg += loss_y.mean()
            loss_naive_avg += loss_naive.mean()
            loss_avg_div += 1
    validate(model.config.train_epochs)


if __name__ == '__main__':
    main()

