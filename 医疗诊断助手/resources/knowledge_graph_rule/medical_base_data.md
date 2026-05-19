# 医疗知识图谱基础信息MD记录文件

## 一、节点类型及其属性

| 节点标签（Label） | 属性（Property） |
| :---------------- | :--------------- |
| **Disease** | `name`, `desc`, `prevent`, `cause`, `easy_get`, `cure_lasttime`, `cure_department`, `cure_way`, `cured_prob` |
| **Drug**    | `name` |
| **Food**    | `name` |
| **Check**   | `name` |
| **Department** | `name` |
| **Producer**   | `name` |
| **Symptom**    | `name` |

> **说明**：除 `Disease` 节点外，其余类型节点仅包含 `name` 属性。

## 二、关系类型及其信息

| 关系类型（rel_type） | 起始节点 → 结束节点 | 关系属性 | 关系含义（rel_name） | 对应数据列表 |
| :------------------- | :------------------ | :------- | :------------------- | :----------- |
| `recommand_eat` | Disease → Food | `name` | 推荐食谱 | `rels_recommandeat` |
| `no_eat`        | Disease → Food | `name` | 忌吃     | `rels_noteat` |
| `do_eat`        | Disease → Food | `name` | 宜吃     | `rels_doeat` |
| `belongs_to`    | Department → Department | `name` | 属于 | `rels_department`（细分科室 → 上级科室） |
| `common_drug`   | Disease → Drug | `name` | 常用药品 | `rels_commonddrug` |
| `drugs_of`      | Producer → Drug | `name` | 生产药品 | `rels_drug_producer` |
| `recommand_drug`| Disease → Drug | `name` | 好评药品 | `rels_recommanddrug` |
| `need_check`    | Disease → Check | `name` | 诊断检查 | `rels_check` |
| `has_symptom`   | Disease → Symptom | `name` | 症状   | `rels_symptom` |
| `acompany_with` | Disease → Disease | `name` | 并发症 | `rels_acompany` |
| `belongs_to`    | Disease → Department | `name` | 所属科室 | `rels_category` |

> **补充说明**：
> - 每个关系在创建时均包含一个 `name` 属性，其值为“关系含义”列中的中文字符串。
> - 关系方向与代码中的 `start_node` → `end_node` 一致。
> - `rels_department` 表示从细分科室指向其上级科室。