import torch



def Tester(net, test_loader, criterion, device):
    net.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    for idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicts = torch.max(outputs.data, 1)
        test_total += targets.size(0)
        test_correct += predicts.eq(targets.data).cpu().sum()

    print("[Test] Loss: %.3f | Acc: %.3f%% (%d / %d)\n" 
            % (test_loss / (idx + 1), 100. * test_correct / test_total, test_correct, test_total))
    

def Trainer(net, train_loader, test_loader, optimizer, scheduler, criterion, n_epochs, print_frq, device):
    for epoch in range(n_epochs):
        print(f"\n[train] Epoch: {epoch}")
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicts = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicts.eq(targets.data).cpu().sum()

            if ((idx + 1) % print_frq == 0):
                print("iteration[%d / %d] Loss: %.3f | Acc: %.3f%% (%d / %d)"
                        % ((idx + 1), len(train_loader), train_loss / (idx + 1), 100. * correct / total, correct, total))
                
        if (epoch % 10 == 0 or epoch == (n_epochs - 1)):
            Tester(net, test_loader, criterion, device)

        scheduler.step()