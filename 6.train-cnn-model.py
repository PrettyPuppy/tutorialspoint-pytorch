def fit(epoch,model,data_loader,phase='training',volatile = False):
   if phase == 'training':
      model.train()
   if phase == 'training':
      model.train()
   if phase == 'validation':
      model.eval()
   volatile=True
   running_loss = 0.0
   running_correct = 0
   for batch_idx , (data,target) in enumerate(data_loader):
      if is_cuda:
         data,target = data.cuda(),target.cuda()
         data , target = Variable(data,volatile),Variable(target)
      if phase == 'training':
         optimizer.zero_grad()
         output = model(data)
         loss = F.nll_loss(output,target)
         running_loss + = 
         F.nll_loss(output,target,size_average = 
         False).data[0]
         preds = output.data.max(dim = 1,keepdim = True)[1]
         running_correct + = 
         preds.eq(target.data.view_as(preds)).cpu().sum()
         if phase == 'training':
            loss.backward()
            optimizer.step()
   loss = running_loss/len(data_loader.dataset)
   accuracy = 100. * running_correct/len(data_loader.dataset)
   print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{return loss,accuracy}})
