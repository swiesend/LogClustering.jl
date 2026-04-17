# Models

Stage C — the compression / encoder-decoder track — and Stage F's
sequential model.

## KATE (modernised K-competitive autoencoder layer)

```@docs
LogClustering.KATE
LogClustering.KATE.KCompetetive
LogClustering.KATE.normalize_log
LogClustering.KATE.count_words
LogClustering.KATE.transform_text_to_input
LogClustering.KATE.get_similar_words
```

## DeepKATE (thesis §3.2.3, ported to Lux)

```@docs
LogClustering.DeepKATE
LogClustering.DeepKATE.deep_kate
LogClustering.DeepKATE.deep_kate_loss
LogClustering.DeepKATE.latent_layer
LogClustering.DeepKATE.repel
```

## SeqLSTM (thesis §3.2.8, ported + bi-dir / peephole)

```@docs
LogClustering.SeqLSTM
LogClustering.SeqLSTM.seq_lstm
LogClustering.SeqLSTM.seq_lstm_loss
LogClustering.SeqLSTM.predict_next
LogClustering.SeqLSTM.PeepholeLSTM
```

## VQ-VAE (van den Oord et al. 2017)

```@docs
LogClustering.VQVAE
LogClustering.VQVAE.VectorQuantizer
LogClustering.VQVAE.vq_vae
LogClustering.VQVAE.vq_vae_loss
LogClustering.VQVAE.assign_codes
```

## SimCSE (Gao, Yao & Chen 2021)

```@docs
LogClustering.SimCSE
LogClustering.SimCSE.simcse_loss
LogClustering.SimCSE.simcse_step_loss
```
