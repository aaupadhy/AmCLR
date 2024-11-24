class SogCLR_Loss_with_Augmentation(nn.Module):
    def __init__(self, N=2900000, gamma=0.1, temperature=0.07, world_size=8, bsz=128, enable_surrogate=False, surrogate_c=1.0,
                lamda_rho=1.0, lamda_init=1.0):
              
        super(SogCLR_Loss_with_Augmentation, self).__init__()
        self.world_size = world_size
        self.s_I = torch.zeros(N).cuda()
        self.s_T = torch.zeros(N).cuda()
        self.b_I = torch.zeros(N).cuda()
        self.b_T = torch.zeros(N).cuda()
        self.gamma = gamma
        self.temperature = temperature
        self.eps = 1e-8
        self.bsz = bsz
        self.enable_surrogate = enable_surrogate
        self.c = surrogate_c

    def _sqh(self, x):
        return torch.max(torch.zeros_like(x), x + self.c) ** 2

    def forward(self, image_features, text_features, augmented_image_features, augmented_text_features, image_ids, text_ids, epoch):
        if self.world_size > 1:
            image_features = torch.cat(GatherLayer.apply(image_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)
            augmented_image_features = torch.cat(GatherLayer.apply(augmented_image_features), dim=0)
            augmented_text_features = torch.cat(GatherLayer.apply(augmented_text_features), dim=0)

        sim_orig = torch.einsum('i d, j d -> i j', image_features, text_features)
        sim_aug_img = torch.einsum('i d, j d -> i j', augmented_image_features, text_features)
        sim_aug_text = torch.einsum('i d, j d -> i j', image_features, augmented_text_features)
        sim_aug_both = torch.einsum('i d, j d -> i j', augmented_image_features, augmented_text_features)

        sim_image = torch.cat([sim_orig, sim_aug_text], dim=1)
        sim_text = torch.cat([sim_orig, sim_aug_img], dim=0)
        sim_both = torch.cat([torch.cat([sim_orig, sim_aug_text], dim=1),
                              torch.cat([sim_aug_img, sim_aug_both], dim=1)], dim=0)

        diag_sim = torch.cat([torch.diagonal(sim_orig), torch.diagonal(sim_aug_both)])

        image_diffs = sim_image - diag_sim[:, None]
        text_diffs = sim_text - diag_sim[None, :]

        mask_positive_image = torch.eye(sim_image.shape[1]).cuda()[:self.bsz] + torch.eye(sim_image.shape[1]).cuda()[self.bsz:]
        mask_positive_text = torch.eye(sim_text.shape[0]).cuda()[:self.bsz] + torch.eye(sim_text.shape[0]).cuda()[self.bsz:]

        mask_negative_image = 1.0 - mask_positive_image
        mask_negative_text = 1.0 - mask_positive_text

        if self.enable_surrogate:
            image_diffs = self._sqh(image_diffs)
            text_diffs = self._sqh(text_diffs)

        image_diffs_d_temps = (image_diffs / self.temperature).clone().detach_()
        text_diffs_d_temps = (text_diffs / self.temperature).clone().detach_()

        old_b_I = self.b_I[image_ids]
        new_b_I = torch.max(image_diffs_d_temps * mask_negative_image, old_b_I[:, None].tile(1, image_diffs_d_temps.shape[1]))
        self.b_I[image_ids] = torch.max(new_b_I, dim=1)[0]

        old_b_T = self.b_T[text_ids]
        new_b_T = torch.max(text_diffs_d_temps * mask_negative_text, old_b_T[None, :].tile(text_diffs_d_temps.shape[0], 1))
        self.b_T[text_ids] = torch.max(new_b_T, dim=0)[0]

        exp_image_diffs = torch.exp(image_diffs_d_temps - self.b_I[image_ids][:, None]) * mask_negative_image
        exp_text_diffs = torch.exp(text_diffs_d_temps - self.b_T[text_ids][None, :]) * mask_negative_text

        g_I = torch.sum(exp_image_diffs, dim=1, keepdim=True) / (exp_image_diffs.shape[1] - 2)
        g_T = torch.sum(exp_text_diffs, dim=0, keepdim=True) / (exp_text_diffs.shape[0] - 2)

        if epoch == 0:
            s_I = g_I
            s_T = g_T
        else:
            s_I = (1.0 - self.gamma) * self.s_I[image_ids] * torch.exp(old_b_I - self.b_I[image_ids]) + self.gamma * g_I.squeeze()
            s_T = (1.0 - self.gamma) * self.s_T[text_ids] * torch.exp(old_b_T - self.b_T[text_ids]) + self.gamma * g_T.squeeze()
            s_I = s_I.reshape(g_I.shape)
            s_T = s_T.reshape(g_T.shape)

        self.s_I[image_ids] = s_I.squeeze()
        self.s_T[text_ids] = s_T.squeeze()

        weights_image = exp_image_diffs / (s_I + self.eps)
        weights_text = exp_text_diffs / (s_T + self.eps)

        if torch.any(torch.isnan(weights_image)):
            assert 0, "weights_image has nan."
        if torch.any(torch.isnan(weights_text)):
            assert 0, "weights_text has nan."

        image_loss = torch.sum(weights_image * image_diffs, dim=1, keepdim=True) / (exp_image_diffs.shape[1] - 2)
        text_loss = torch.sum(weights_text * text_diffs, dim=0, keepdim=True) / (exp_text_diffs.shape[0] - 2)

        total_loss = image_loss.mean() + text_loss.mean()

        return total_loss, 0.0, 0.0
