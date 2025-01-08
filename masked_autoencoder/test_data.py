import torch 

TEST_DATA = {}
TEST_DATA['inputs'] = torch.randn(3, 3, 112, 112)
TEST_DATA['inputs_channel_idx'] = torch.randint(0, 3, (3, 3))

