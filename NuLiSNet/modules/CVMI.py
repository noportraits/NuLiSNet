import torch.nn as nn
import torch
from modules.CSM import CSM


class CVMI(nn.Module):

    def __init__(self, channels):
        super(CVMI, self).__init__()
        self.fe = CSM(channels)
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()

    def forward(self, low1, low2):
        Q, K = self.conv(self.fe(low1)), self.conv(self.fe(low2))
        b, c, h, w = Q.shape
        score = torch.bmm(Q.permute(0, 2, 3, 1).contiguous().view(-1, w, c),
                          K.permute(0, 2, 1, 3).contiguous().view(-1, c, w))
        left = low1 + torch.bmm(self.relu(self.softmax(score)),
                                low2.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w,
                                                                                                        c).permute(0,
                                                                                                                   3,
                                                                                                                   1,
                                                                                                                   2)
        right = low2 + torch.bmm(self.relu(self.softmax(score.permute(0, 2, 1))),
                                 low1.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w,
                                                                                                         c).permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    2)

        return left, right


class New_CVMI(nn.Module):

    def __init__(self, channels):
        super(New_CVMI, self).__init__()
        self.fe1 = CSM(channels)
        self.fe2 = CSM(channels)
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()

    def forward(self, low1, low2):
        Q1, K1 = self.conv(self.fe1(low1)), self.conv(self.fe1(low2))
        Q2, K2 = self.conv(self.fe2(low1)), self.conv(self.fe2(low2))
        b, c, h, w = Q1.shape
        score1 = torch.bmm(Q1.permute(0, 2, 3, 1).contiguous().view(-1, w, c),
                          K1.permute(0, 2, 1, 3).contiguous().view(-1, c, w))
        score1 = self.relu(self.softmax(score1))
        score2 = torch.bmm(Q2.permute(0, 2, 3, 1).contiguous().view(-1, w, c),
                          K2.permute(0, 2, 1, 3).contiguous().view(-1, c, w))
        score2 = self.relu(self.softmax(score2))
        right2left1 = torch.bmm(score1,
                                low2.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w,
                                                                                                        c).permute(0,
                                                                                                                   3,
                                                                                                                   1,
                                                                                                                   2)
        right2left2 = torch.bmm(score2,
                                low2.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w,
                                                                                                        c).permute(0,
                                                                                                                   3,
                                                                                                                   1,
                                                                                                                   2)
        left2right1 = torch.bmm(score1.permute(0, 2, 1),
                                 low1.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w,
                                                                                                         c).permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    2)
        left2right2 = torch.bmm(score2.permute(0, 2, 1),
                                 low1.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w,
                                                                                                         c).permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    2)
        left2right2left = torch.bmm(score2,
                                 left2right1.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w,
                                                                                                         c).permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    2)
        right2left2right = torch.bmm(score2.permute(0, 2, 1),
                                 right2left1.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w,
                                                                                                         c).permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    2)
        left = low1 + right2left1 + right2left2 + left2right2left
        right = low2 + left2right1 + left2right2 + right2left2right

        return left, right
class New_CVMI_2(nn.Module):

    def __init__(self, channels):
        super(New_CVMI_2, self).__init__()
        self.fe1 = CSM(channels)
        self.fe2 = CSM(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.softmax = nn.Softmax(-1)
        self.relu = nn.ReLU()

    def forward(self, low1, low2):
        Q1, K1 = self.conv1(self.fe1(low1)), self.conv1(self.fe1(low2))
        Q2, K2 = self.conv2(self.fe2(low1)), self.conv2(self.fe2(low2))
        b, c, h, w = Q1.shape
        score1 = torch.bmm(Q1.permute(0, 2, 3, 1).contiguous().view(-1, w, c),
                          K1.permute(0, 2, 1, 3).contiguous().view(-1, c, w))
        score1 = self.relu(self.softmax(score1))
        score2 = torch.bmm(Q2.permute(0, 2, 3, 1).contiguous().view(-1, w, c),
                          K2.permute(0, 2, 1, 3).contiguous().view(-1, c, w))
        score2 = self.relu(self.softmax(score2))
        right2left1 = torch.bmm(score1,
                                low2.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w,
                                                                                                        c).permute(0,
                                                                                                                   3,
                                                                                                                   1,
                                                                                                                   2)
        right2left2right = torch.bmm(score2.permute(0, 2, 1),
                                right2left1.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w,
                                                                                                        c).permute(0,
                                                                                                                   3,
                                                                                                                   1,
                                                                                                                   2)
        left2right2 = torch.bmm(score2.permute(0, 2, 1),
                                 low1.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w,
                                                                                                         c).permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    2)
        left2right2left = torch.bmm(score1,
                                left2right2.permute(0, 2, 3, 1).contiguous().view(-1, w, c)).contiguous().view(b, h, w,
                                                                                                        c).permute(0,
                                                                                                                   3,
                                                                                                                   1,
                                                                                                                   2)

        left = low1 + right2left1 + left2right2left
        right = low2 + left2right2 + right2left2right

        return left, right