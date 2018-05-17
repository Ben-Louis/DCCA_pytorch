
class CCA_torch:
    def __init__(self, output_dim=1, use_all_singular_values=True, cca_space_dim=10, device=None):
        self.output_dim = output_dim
        self.cca_space_dim = cca_space_dim
        self.use_all_singular_values = use_all_singular_values
        self.device = device
    
    def call(self, x):
        r1 = r2 = 1e-4
        eps = 1e-12

        o1 = o2 = m = x.size(1) // 2
        H1, H2 = x[:, :m], x[:, m:]

        H1bar = H1 - torch.mean(H1, dim=0).view(1,-1)
        H2bar = H2 - torch.mean(H2, dim=0).view(1,-1)

        SigmaHat12 = torch.t(H1bar).mm(H2bar) / (m-1)
        SigmaHat11 = torch.t(H1bar).mm(H1bar) / (m-1) 
        SigmaHat22 = torch.t(H2bar).mm(H2bar) / (m-1) 
        SigmaHat11 += r1 * torch.diag(torch.ones(o1)).type(x.type())
        SigmaHat22 += r2 * torch.diag(torch.ones(o2)).type(x.type())     


        [D1, V1] = torch.symeig(SigmaHat11, True)
        [D2, V2] = torch.symeig(SigmaHat22, True)
    

        D1_indices = (D1>eps)
        D2_indices = (D2>eps)

        D1 = D1[D1_indices]
        D2 = D2[D2_indices]    
        tmp = torch.FloatTensor([1]*V1.size(1)).to(self.device) 
        
        D1_indices = D1_indices.view(-1,1).float().matmul(tmp.view(1,-1)).t().byte()
        D2_indices = D2_indices.view(-1,1).float().matmul(tmp.view(1,-1)).t().byte()


        V1 = V1[D1_indices].view(V1.size(1), -1)
        V2 = V2[D2_indices].view(V2.size(1), -1)    

    
        SigmaHat11RootInv = (V1.matmul(torch.diag(D1.pow(-0.5)))).matmul(V1.transpose(0,1))
        SigmaHat22RootInv = (V2.matmul(torch.diag(D2.pow(-0.5)))).matmul(V2.transpose(0,1))

        Tval = SigmaHat11RootInv.matmul(SigmaHat12).matmul(SigmaHat22RootInv)
        U, S, V = torch.svd(Tval, True)

        corr = torch.sum(S)
        #A1star = SigmaHat11RootInv.mm(U)
        #A2star = SigmaHat22RootInv.mm(V)

        delta12 = SigmaHat11RootInv.mm(U).mm(torch.t(V)).mm(SigmaHat22RootInv)
        delta11 = - 0.5 * SigmaHat11RootInv.mm(U).mm(torch.diag(S)).mm(torch.t(U)).mm(SigmaHat11RootInv)
        delta22 = - 0.5 * SigmaHat22RootInv.mm(V).mm(torch.diag(S)).mm(torch.t(V)).mm(SigmaHat22RootInv) #???

        grad1 = (2*H1bar.mm(delta11) + H2bar.mm(torch.t(delta12))) / (m-1)
        grad2 = (2*H2bar.mm(delta22) + H1bar.mm(delta12)) / (m-1)
        grad = torch.cat([grad1, grad2], dim=1)
    
        return corr, grad
  
  
    def __call__(self, x):
        return self.call(x)
