# introduce
This brach is based on commit fa1ac1f6a4e21c667df1fb6344ab19658f5d3e1b, target is support continous batching.
Now support CPU Attention version.
# TODO List
- Modify to new op `PackedAttention`, quant situation, special mask
- Modify to additional thread waiting req and response function.
- Remove kvcache problem: kvcache will release in next attention call.
- BatchScheduler algorithm
- Mask pos embedding TODO
- KVCacheManager: prefix name and prefix cahce dir modify
- CPUKVCacheManager:  verify quant
- AttentionTest
- other backend support