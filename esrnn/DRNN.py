import torch 
import torch.nn as nn 
import torch.autograd as autograd 


use_cuda = torch.cuda.is_available()
class DRNN(nn.Module):
    
    def __init__(self,n_input,n_hidden,n_layers,
                      dilations,dropout=0,cell_type='GRU',
                      batch_first=False):
    
        super(DRNN,self).__init__()

        self.dilations = dilations
        self.cell_type = cell_type
        self.batch_first = batch_first

        if self.cell_type == "GRU":
            cell = nn.GRU
        elif self.cell_type == "RNN":
            cell = nn.RNN
        elif self.cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError
        
        layers = []
        for i in range(n_layers):
            if i == 0:
                c = cell(n_input,n_hidden,dropout = dropout)
            else:
                c = cell(n_hidden,n_hidden,dropout = dropout)
            
            layers.append(c)
        
        self.cells = nn.Sequential(*layers)

    def forward(self,inputs,hidden=None):
        if self.batch_first:
            inputs = input.trainspose(0,1)
        
        outputs = []

        for i,(cell,dilations) in enumerate(zip(self.cells,self.dilations)):

            if hidden is None:
                inputs, _ = self.drnn_layer(cell, inputs, dilations)

            outputs.append(inputs[-dilations:])

        if self.batch_first:
            inputs = input.trainspose(0,1) 

        return inputs,outputs
    
    def drnn_layer(self,cell,inputs,rate,hidden = None):
        
        n_steps = len(inputs)
        batch_size = inputs[0].size(0)
        hidden_size = cell.hidden_size
        
        inputs,dilated_steps = self._pad_inputs(inputs,n_steps,rate)
        dilated_inputs = self._prepare_inputs(inputs, rate)
        
        
        if hidden is None:
            dilated_outputs,hidden = self._apply_cell(dilated_inputs, cell, batch_size, rate, hidden_size)

        splitted_outputs = self._split_outputs(dilated_outputs, rate)
        outputs = self._unpad_outputs(splitted_outputs, n_steps)

        return outputs, hidden

    def _pad_inputs(self,inputs,n_steps,rate):
        
        iseven = (n_steps %rate) == 0         

        dilated_steps  = None
        if not iseven :
            dilated_steps = n_steps //rate + 1
            # 26;dilated_steps*rate-inputs.size(0) = 2
            zeros_  = torch.zeros(dilated_steps*rate-inputs.size(0),
                                inputs.size(1),
                                inputs.size(2))
            if use_cuda:
                zeros_ = zeros_.cuda()

            inputs = torch.cat((inputs,autograd.Variable(zeros_)))
        else :
            dilated_steps = n_steps//rate

        return inputs,dilated_steps
        
    def _prepare_inputs(self,inputs,rate):
        dilated_inputs = torch.cat([inputs[j::rate,:,:] for j in range(rate)],1)
        return dilated_inputs

    def _apply_cell(self,dilated_inputs,cell,batch_size,rate,hidden_size,hidden = None):
        
        if hidden is None:
            if self.cell_type == "LSTM":
                c,m = self.init_hidden(batch_size*rate,hidden_size)
                hidden = (c.unsqueeze(0),m.unsqueeze(0))
            else :
                hidden = self.init_hidden(batch_size*rate,hidden_size).unsqueeze(0)
                
        dilated_outputs, hidden = cell(dilated_inputs, hidden)

        return dilated_outputs, hidden

    def init_hidden(self,batch_size,hidden_dim):

        hidden = autograd.Variable(torch.zeros(batch_size,hidden_dim))
        
        if use_cuda :
            hidden = hidden.cuda()
        
        if self.cell_type == "LSTM":
            memory = autograd.Variable(torch.zeros(batch_size,hidden_dim))
            if use_cuda:
                memory = memory.cuda()
            
            return hidden, memory
        else:
            return hidden
    
    def _split_outputs(self,dilated_outputs,rate):
        batchsize = dilated_outputs.size(1) // rate

        blocks = [dilated_outputs[:,i*batchsize:(i+1)*batchsize,:] for i in range(rate)]

        interleaved = torch.stack((blocks)).transpose(1,0).contiguous()
        interleaved = interleaved.view(dilated_outputs.size(0) * rate,
                                       batchsize,
                                       dilated_outputs.size(2))
        return interleaved
    
    def _unpad_outputs(self,splitted_outputs,n_steps):
        return splitted_outputs[:n_steps]

        
if __name__ == "__main__":
    n_input = 13
    n_hidden = 50
    n_layers = 2
    cell_type = 'LSTM'
    dilations = (1,7)

    model = DRNN(n_input, n_hidden, n_layers,dilations, cell_type=cell_type)

    x1 = torch.randn(180, 1024, n_input)
    x2 = torch.randn(23, 2, n_input)

    out, hidden = model(x1.to('cuda'))
    #out, hidden = model(x2, hidden)

