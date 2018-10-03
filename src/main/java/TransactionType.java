public enum TransactionType {
    PAYMENT(1),
    TRANSFER(2),
    CASH_OUT(3),
    DEBIT(4),
    CASH_IN(5);

    private int transactionType;

    TransactionType(int transactionType) {

        this.transactionType = transactionType;
    }

    public double getTransactionType() {
        return transactionType;
    }
}
