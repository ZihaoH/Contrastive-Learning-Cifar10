from tqdm import tqdm
from utils import *
import time
from .train import test

logger = logging.getLogger(__name__)

def adv_train_loop(args, model, criterion, optimizer, scheduler, trainloader, testloader):
    logger.warning("***** Running Training *****")
    logger.warning(f"   Total steps = {args.total_steps}")

    data_iter = iter(trainloader)

    for step in range(args.start_step, args.total_steps):
        model.set_mixbn(True)
        if step % args.eval_step == 0:
            pbar = tqdm(range(args.eval_step))
            batch_time = AverageMeter()
            data_time = AverageMeter()
            train_losses = AverageMeter()


        model.train()
        end = time.time()

        try:
            images, targets = data_iter.next()
        except:
            data_iter = iter(trainloader)
            images, targets = data_iter.next()

        data_time.update(time.time() - end)

        images = images.cuda()
        targets = targets.cuda()
        tmp_targets = targets

        outputs, targets = model(images, targets)

        outputs = outputs.transpose(1, 0).contiguous().view(-1, args.num_classes)
        targets = targets.transpose(1, 0).contiguous().view(-1)

        loss = criterion(outputs, targets)
        loss.backward()
        if args.optim != 'sam':
            optimizer.step()
        else:
            optimizer.first_step(zero_grad=True)
            # second forward-backward pass
            second_outputs, targets = model(images, tmp_targets)
            second_outputs = second_outputs.transpose(1, 0).contiguous().view(-1, args.num_classes)
            targets = targets.transpose(1, 0).contiguous().view(-1)
            criterion(second_outputs, targets).backward()  # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)

        optimizer.zero_grad()
        scheduler.step()

        train_losses.update(loss.item())
        batch_time.update(time.time() - end)

        # a=accuracy(outputs[:128],targets[:128])
        # b=accuracy(outputs[128:], targets[128:])
        # c = accuracy(outputs, targets)
        # print('normal: {}, attack: {}, all: {}'.format(a,b,c))

        pbar.set_description(
            f"Train Iter: {step + 1:3}/{args.total_steps:3}. "
            f"LR: {get_lr(optimizer):.4f}. Data: {data_time.avg:.2f}s. "
            f"Batch: {batch_time.avg:.2f}s. Train_Loss: {train_losses.avg:.4f}. ")
        pbar.update()

        args.writer.add_scalar("lr", get_lr(optimizer), step)
        args.num_eval = step // args.eval_step


        if (step+1) % args.eval_step == 0:
            pbar.close()
            args.writer.add_scalar("train/1.train_losses", train_losses.avg, args.num_eval)

            test_loss, top1 = test(model, testloader, criterion)

            args.writer.add_scalar("test/loss", test_loss, args.num_eval)
            args.writer.add_scalar("test/acc@1", top1, args.num_eval)
            visualize_param_hist(args.writer, model, (step+1) // args.eval_step)

            is_best = top1 > args.best_top1
            if is_best:
                args.best_top1 = top1
            logger.warning(f"top-1 acc: {top1:.2f}, Best top-1 acc: {args.best_top1:.2f}")

            save_checkpoint(args, {
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'best_top1': args.best_top1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best)


def test(model, testloader, criterion):
    model.set_mixbn(False)
    model.train(False)
    correct = 0
    total = 0
    loss_all = 0

    with torch.no_grad():
        for test_x, test_y in testloader:
            images, labels = test_x.cuda(), test_y.cuda()
            output,labels = model(images, labels)
            loss = criterion(output, labels)
            loss_all += loss.item()*images.shape[0]
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    loss_avg = loss_all / total
    return loss_avg, accuracy