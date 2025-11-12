
## **Training vs. Evaluation Modes**

In PyTorch:

* **`model.train()`** → enables **training mode**:

  * Activates dropout, batchnorm updates, etc.
  * Gradients are tracked.

* **`model.eval()`** → enables **evaluation mode**:

  * Disables dropout, batchnorm uses running stats.
  * Gradients are not tracked if you also wrap code in `torch.no_grad()` or `torch.inference_mode()`.

> These modes control **how the model behaves**, not *where the code is physically placed*.

---

## ** Can evaluation be in a separate section?**

Yes, absolutely

You **don’t have to evaluate inside the same loop iteration as training**. There are two common patterns:

---

### **Option A — Evaluate every epoch inside the loop**

```python
for epoch in range(epochs):
    model.train()
    # training code
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

    # evaluation in the same loop
    model.eval()
    with torch.inference_mode():
        y_pred_test = model(X_test)
        test_loss = loss_fn(y_pred_test, y_test)
```

Advantage: Can track training + test metrics **epoch by epoch**.

---

### **Option B — Evaluate in a separate loop / later**

```python
# Train model for all epochs
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

# After training, evaluate on test set
model.eval()
with torch.inference_mode():
    y_pred_test = model(X_test)
    test_loss = loss_fn(y_pred_test, y_test)
    y_pred_classes = torch.argmax(y_pred_test, dim=1)
```

Advantage: Simplifies code if you only care about final performance.

---

## Important Notes

1. **`model.eval()` is independent of training iteration**

   * You can switch to eval whenever you want.
   * Must remember to switch back to `model.train()` if you continue training.

2. **Wrap eval in `torch.inference_mode()`**

   * Prevents gradient tracking → saves memory and speeds up evaluation.

3. **Shuffling**

   * Training data can still be shuffled each epoch, independent of evaluation.

---

### **Summary**

* `eval()` **does not need to be inside the same loop as training**.
* Common patterns:

  * Evaluate every epoch inside loop → monitor performance.
  * Evaluate only at the end → simpler code.
* Must always switch to `model.train()` before the next training batch if continuing training.

