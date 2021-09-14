import pandas as pd
import plotnine as p9

# From 08-31 data
data = {
        '-type -badwords': [1.099, 1.008, 1.004, 1.006],
        '+type +badwords': [0.832, 0.843, 0.772, 1.087],
        '+type -badwords': [1.023, 1.331, 0.878, 0.885],
}

df = pd.DataFrame(data)
df *= 100  # To pct
df = df.stack().reset_index().rename(columns={0: 'pct_of_naive'})

p = (
        p9.ggplot(df)
        #+ p9.geom_boxplot(p9.aes(x='factor(level_1)', y='pct_of_naive'))
        + p9.geom_violin(p9.aes(x='factor(level_1)', y='pct_of_naive'))
        + p9.scale_x_discrete(name='Data')
        + p9.scale_y_continuous(name='Mean-squared-error, % of naive (lower is better)')
        )
p.save('test.png')

