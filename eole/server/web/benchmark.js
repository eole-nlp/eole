/* Fixed workloads and reporting shared by the benchmark UI and tests. */
window.EoleBenchmark = (() => {
  const profiles = [
    {
      id: "base",
      label: "Base · native heads",
      speculation: "off",
      vocabulary_quantization: "none",
      dflash2_quantization: "none",
    },
    {
      id: "mtp-native",
      label: "MTP · native heads",
      speculation: "mtp",
      vocabulary_quantization: "none",
      dflash2_quantization: "none",
    },
    {
      id: "mtp-int4",
      label: "MTP · INT4 heads",
      speculation: "mtp",
      vocabulary_quantization: "int4",
      dflash2_quantization: "none",
    },
    {
      id: "dflash-native",
      label: "DFlash2 · BF16 drafter / native heads",
      speculation: "dflash2",
      vocabulary_quantization: "none",
      dflash2_quantization: "none",
    },
    {
      id: "dflash-int8",
      label: "DFlash2 · INT8 drafter / native heads",
      speculation: "dflash2",
      vocabulary_quantization: "none",
      dflash2_quantization: "int8",
    },
    {
      id: "dflash-int8-int4",
      label: "DFlash2 · INT8 drafter / INT4 heads",
      speculation: "dflash2",
      vocabulary_quantization: "int4",
      dflash2_quantization: "int8",
    },
  ];
  const workloads = [
    {
      id: "code",
      label: "Python code",
      messages: [
        {
          role: "system",
          content:
            "Write practical Python software with clear interfaces and executable examples.",
        },
        {
          role: "user",
          content:
            "Build a Python 3.11 command-line tool that inventories a photo archive. Traverse subdirectories, compute SHA-256 checksums to identify duplicate files, and store paths, sizes, and modification timestamps in SQLite. Support incremental rescans, a dry-run cleanup report, and CSV export. Never delete files automatically. Include error handling for unreadable files and paths containing Unicode, a small test suite using temporary directories, and installation instructions. Show the full contents of each file in labeled code blocks.",
        },
      ],
    },
    {
      id: "english",
      label: "English story",
      messages: [
        {
          role: "system",
          content:
            "Write immersive fiction with distinct characters, concrete details, and a coherent resolution. Return the story alone.",
        },
        {
          role: "user",
          content:
            "Write a mystery of roughly 2,000 words in five named chapters. At a coastal museum the morning after a storm, a conservator discovers that a recently restored painting has been replaced by an unfinished copy. A ferry operator, a retired teacher, and the museum's night attendant each know part of what happened. Work a torn tide timetable, a smear of blue pigment, and a misplaced hearing aid into the investigation. Introduce the decisive evidence before the solution. Explain how the substitution occurred and why the person responsible believed it would help someone. End with the conservator deciding what to tell the museum board.",
        },
      ],
    },
    {
      id: "translation",
      label: "Chinese → English",
      messages: [
        {
          role: "user",
          content:
            "Translate this Chinese project brief into fluent English for a municipal planning team. Retain the numbered headings, quantities, deadlines, and distinctions between proposals and commitments. Follow it with a bilingual glossary of twelve planning terms.\n\n一、项目背景。旧火车站旁的仓库已经闲置八年。市政府计划将其改造成社区图书馆和维修工坊，但尚未批准最终预算。居民希望保留砖墙和屋顶木梁，同时改善轮椅通行条件。附近学校提出，每周至少安排两个下午供学生参加免费活动。\n\n二、场地调查。初步检查发现东侧墙面渗水，地下排水管也需要更换。结构工程师将在六月十五日前提交报告。在报告完成之前，不得拆除承重构件。院内有三棵成熟的树，施工围挡应避开根系保护范围。\n\n三、空间安排。首层拟设置阅览区、儿童活动室和公共卫生间。维修工坊需要独立通风，以免噪声和粉尘影响阅读。二层是否开放取决于消防疏散评估；如果暂时关闭，可以先作为设备维护空间。设计团队应比较两种方案的建设成本与十年运行费用。\n\n四、公众参与。社区将在七月举行两场讨论会，并提供线上意见表。问卷结果需要按年龄组汇总，但不得公开个人联系方式。对于无法到场的居民，工作人员可以预约电话访谈。意见报告应明确区分多数建议、少数意见和法律要求。\n\n五、采购与实施。工程拟分为防水修缮、室内改造和设备安装三个阶段。招标文件应要求承包商说明材料来源及保修条件。若投标价格超过预算，应先调整非必要装修，不应削减无障碍设施。所有工期均为暂定，正式开工日期将在许可证批准后公布。\n\n六、验收与运营。开放前需要完成空气质量检测和消防检查。首年的使用人数、维修活动次数及能源消耗将按季度公布。运营团队还应记录居民提出的问题及处理进度，以便下一年度评估服务安排。",
        },
      ],
    },
  ];
  function available(profile, model) {
    const options = model?.speculation ?? {};
    return (
      profile.speculation === "off" ||
      (profile.speculation === "mtp"
        ? options.mtp_available
        : options.dflash2_available)
    );
  }
  function median(values) {
    if (!values.length) return null;
    const a = values.slice().sort((x, y) => x - y),
      middle = Math.floor(a.length / 2);
    return a.length % 2 ? a[middle] : (a[middle - 1] + a[middle]) / 2;
  }
  function summarize(runs, limit, expectedMode, repeats) {
    const actualMode = (run) =>
      run.speculation_mode ?? (run.mtp_active ? "mtp" : "off");
    const valid = runs.filter(
      (run) =>
        !run.cancelled &&
        run.output_tokens === limit &&
        actualMode(run) === expectedMode &&
        Number.isFinite(run.decode_tokens_per_second),
    );
    const complete = valid.length === repeats;
    return {
      valid_runs: valid.length,
      measured_runs: runs.length,
      status: complete ? "complete" : "incomplete",
      decode_tokens_per_second: complete
        ? median(valid.map((r) => r.decode_tokens_per_second))
        : null,
      first_token_seconds: complete
        ? median(
            valid.map((r) => r.first_token_seconds).filter(Number.isFinite),
          )
        : null,
      acceptance:
        complete && expectedMode !== "off"
          ? median(valid.map((r) => r.mtp_acceptance).filter(Number.isFinite))
          : null,
    };
  }
  return { profiles, workloads, available, summarize };
})();
