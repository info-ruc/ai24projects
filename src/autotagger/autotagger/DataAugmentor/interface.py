class DataAugmentorInterface(object):
    def __init__(self):
        pass

    def fit(self) -> any:
        pass

    def predict(self) -> any:
        pass

    def fit_predict(self) -> any:
        pass

    def __str__(self):
        pass


def main() -> None:
    raise RuntimeError('该方法仅作调试用，该脚本不可调用')


if __name__ == '__main__':
    main()