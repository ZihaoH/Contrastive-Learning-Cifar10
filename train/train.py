from tqdm import tqdm
from utils import *
import time
from regularization import mixup_data, mixup_criterion

logger = logging.getLogger(__name__)

def train_loop(args, model, criterion, optimizer, scheduler, trainloader, testloader):
    logger.warning("***** Running Training *****")
    logger.warning(f"   Total steps = {args.total_steps}")

    data_iter = iter(trainloader)

    for step in range(args.start_step,args.total_steps):
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

        if args.use_mixup:
            images, targets_a, targets_b, lam = mixup_data(images, targets, args.alpha)
            images, targets_a, targets_b = map(Variable, (images, targets_a, targets_b))

        logits = model(images)

        if not args.use_mixup:
            loss = criterion(logits, targets)
        else:
            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)

        loss.backward()
        if args.optim != 'sam':
            optimizer.step()
        else:
            optimizer.first_step(zero_grad=True)
            # second forward-backward pass
            if not args.use_mixup:
                criterion(model(images), targets).backward()
            else:
                mixup_criterion(criterion, model(images), targets_a, targets_b, lam).backward()
            # criterion(model(images), targets).backward()  # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)

        # visualize_param_grad_scalar(args.writer, model, (step + 1) )

        optimizer.zero_grad()
        scheduler.step()

        train_losses.update(loss.item())
        batch_time.update(time.time() - end)

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
    model.train(False)
    correct = 0
    total = 0
    loss_all = 0

    with torch.no_grad():
        for test_x, test_y in testloader:
            images, labels = test_x.cuda(), test_y.cuda()
            output = model(images)
            loss = criterion(output, labels)
            loss_all += loss.item()*images.shape[0]
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    loss_avg = loss_all / total
    return loss_avg, accuracy

