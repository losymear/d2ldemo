from collections import Counter
from typing import List


class Solution:
    def countServers(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        rowC, colC = Counter(), Counter()
        for i in range(m):
            for j in range(n):
                rowC[i] += grid[i][j]
                colC[j] += grid[i][j]

        return sum(1 if rowC[i] >= 2 or colC[j] >= 2 else 0 for i in range(m) for j in range(n))


print(Solution().countServers([[1, 0], [0, 1]]))
