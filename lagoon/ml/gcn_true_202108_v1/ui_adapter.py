"""Adapter for UI interaction.
"""

from .data import BatchManager
from .model import Model
from lagoon.ui.plugin import Plugin

import arrow
import base64
import io
import pandas as pd
import plotnine as p9
import torch

class GcnUiAdapter(Plugin):
    """Use as:

        ./lagoon_cli.py ui lagoon/ml/gcn_true_202108_v1/ui_adapter.py:path/to/model

    """
    def __init__(self, model_file):
        super().__init__(None)

        self.entities_for_training = set()
        if model_file:
            model_saved = torch.load(model_file)
            hparams = model_saved['hparams']
            state_dict = model_saved['state_dict']
            ents = model_saved['entities']
            self.entities_for_training.update(ents[:len(ents) * 9 // 10])
        else:
            hparams = {}
            state_dict = None

        self.model = Model.argparse_create(hparams)
        self.model.build()
        state_dict and self.model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        # NOTE this technically uses a bunch of memory depending on plugin order.
        # Would be better to isolate the multiprocessing pool used by
        # BatchManager
        cfg = self.model.config
        self.bm = BatchManager(batch_size=cfg.batch_size,
                embed_size=cfg.embed_size,
                window_size=cfg.window_size,
                hops=cfg.hops,
                data_badwords=cfg.data_badwords,
                data_cheat=cfg.data_cheat,
                data_type=cfg.data_type,
                precache=0)

        self.w_min = arrow.get('1998-01-01')
        self.w_max = arrow.get('2021-12-31')


    def plugin_details_entity(self, entity):
        """Given some entity (with attributes loaded), return some HTML which
        details the entity.
        """
        windows = torch.arange(self.w_min.timestamp(), self.w_max.timestamp(),
                self.model.config.window_size)
        batches = self.bm.run_windows(self.model, self.model.type_embeds,
                entity.id, windows.tolist())
        data = {'x': [arrow.get(w).datetime for w in windows.tolist()],
                'pred': batches[:, 0].tolist(),
                'true': batches[:, 1].tolist()}
        def date_labels(breaks):
            return[b.strftime('%Y-%m') for b in breaks]
        plt = (
                p9.ggplot(pd.DataFrame(data).melt('x',
                    var_name='type', value_name='y'))
                + p9.aes('x')
                + p9.scales.scale_x_datetime(labels=date_labels)
                + p9.geom_line(p9.aes(y='y', color='type')))
        buf = io.BytesIO()
        plt.save(buf)

        was_training = entity.id in self.entities_for_training

        img_data = base64.b64encode(buf.getbuffer()).decode()
        return f'<div>Used for training? {was_training}</div><div><img src="data:image/png;base64,{img_data}" /></div>'

